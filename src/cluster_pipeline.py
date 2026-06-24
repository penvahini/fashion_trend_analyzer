# src/cluster_pipeline.py
"""
Embeds this run's segmented outfit images with CLIP, then assigns them to
trend lineages: first via src/cluster_registry.py (match against existing
lineages from prior runs), then KMeans for whatever's left over (genuinely
new clusters). See cluster_registry.py's module docstring for why this
replaced "independently KMeans every run, then guess which clusters match."
"""
from pathlib import Path
import os, shutil, argparse, json
from datetime import date
import numpy as np
import pandas as pd
import cv2
import torch
import open_clip
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from observability import track_run
import cluster_registry as registry_mod

PROJECT_ROOT     = Path(__file__).resolve().parents[1]
SEGMENTED_IMAGES_DIR = PROJECT_ROOT / "images" / "segmented_images"
CLUSTERED_IMAGES_DIR = PROJECT_ROOT / "images" / "clustered_images"
RUNS_DIR         = PROJECT_ROOT / "data" / "runs"

IMG_SIZE = 224  # matches both ViT-B/32 (generic CLIP) and Marqo-FashionCLIP's ViT-B/16
CLIP_MODEL_NAME = "hf-hub:Marqo/marqo-fashionCLIP"
# Fashion-tuned CLIP instead of generic openai/CLIP: generic CLIP was trained on
# everything (people, objects, scenes), so it's sensitive to lighting/pose/
# background as well as garment style. A model fine-tuned specifically on
# fashion product images should cluster more on genuinely style-relevant axes
# (silhouette, fabric, garment type) and less on incidental photography
# variance. Loaded via open_clip's hf-hub support rather than `transformers`'
# CLIPModel/CLIPProcessor classes -- the latter crash in this environment
# (a real, reproducible bus error in the vision-tower forward pass and a
# separate mutex crash in CLIPImageProcessor, both checkpoint- and
# environment-specific, not a code bug here) -- open_clip's loading path
# doesn't hit either issue and needs no preprocessing changes since this
# checkpoint uses the same 224x224 size and CLIP normalization constants as
# the model it replaces.

def load_images(dir_path: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted([p for p in dir_path.iterdir() if p.suffix.lower() in exts])
    if not files:
        raise RuntimeError(f"No images in {dir_path}. Did you run segmentation?")
    return files

def preprocess_bgr_to_tensor(bgr, size=IMG_SIZE):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(rgb).permute(2,0,1).float() / 255.0
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3,1,1)
    x = (x - mean) / std
    return x

def embed_clip(files, device="cuda" if torch.cuda.is_available() else "cpu", batch_size=32):
    model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL_NAME)
    model.eval().to(device)

    feats, names = [], []
    with torch.no_grad():
        batch = []
        for p in files:
            img = cv2.imread(str(p))
            if img is None:
                continue
            x = preprocess_bgr_to_tensor(img).unsqueeze(0)
            batch.append(x); names.append(p.name)
            if len(batch) == batch_size:
                xb = torch.cat(batch, dim=0).to(device)
                f = model.encode_image(xb)
                f = torch.nn.functional.normalize(f, dim=1)
                feats.append(f.cpu().numpy()); batch = []
        if batch:
            xb = torch.cat(batch, dim=0).to(device)
            f = model.encode_image(xb)
            f = torch.nn.functional.normalize(f, dim=1)
            feats.append(f.cpu().numpy())
    Z = np.concatenate(feats, axis=0)
    return Z, names

def discover_new_lineages(Z_unmatched, k, do_pca, variance):
    """
    Runs KMeans on the subset of this run's images that didn't match any
    existing lineage, to discover brand-new trends. Returns (sub_labels,
    centroids_raw, counts) -- centroids are computed in raw CLIP space
    (mean of the actual embeddings in each new sub-cluster), not PCA space,
    so they stay comparable to registry centroids in future runs.
    """
    n = Z_unmatched.shape[0]
    if n == 1:
        return np.array([0]), Z_unmatched.copy(), [1]

    k_new = max(1, min(k, n))
    Zs = StandardScaler().fit_transform(Z_unmatched)
    if do_pca and n > 2:
        pca_full = PCA(n_components=min(n - 1, Zs.shape[1])).fit(Zs)
        n_comp = max(1, int(np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= variance)) + 1)
        Zr = PCA(n_components=min(n_comp, n - 1)).fit_transform(Zs)
    else:
        Zr = Zs

    km = KMeans(n_clusters=k_new, random_state=0, n_init=10)
    sub_labels = km.fit_predict(Zr)

    centroids_raw, counts = [], []
    for c in range(k_new):
        mask = sub_labels == c
        centroids_raw.append(Z_unmatched[mask].mean(axis=0))
        counts.append(int(mask.sum()))
    return sub_labels, np.stack(centroids_raw), counts

def cluster_and_export(filenames, Z, k, run_id, seg_dir, run_out_dir, clustered_images_dir, do_pca=True, variance=0.90):
    reg = registry_mod.load_registry()
    assigned, unmatched_mask = registry_mod.assign_to_existing_lineages(Z, reg)

    n_matched_existing = int((~unmatched_mask).sum())
    n_new_lineages = 0
    if unmatched_mask.any():
        unmatched_idxs = np.where(unmatched_mask)[0]
        sub_labels, new_centroids, new_counts = discover_new_lineages(Z[unmatched_idxs], k, do_pca, variance)
        new_lineage_ids = registry_mod.add_new_lineages(reg, new_centroids, new_counts, run_id)
        n_new_lineages = len(new_lineage_ids)
        for local_idx, sub_label in zip(unmatched_idxs, sub_labels):
            assigned[local_idx] = new_lineage_ids[sub_label]

    # Only update centroids for images that matched a *pre-existing* lineage --
    # newly-created lineages already got their correct initial centroid/count
    # from discover_new_lineages, so re-running update here would double-count them.
    matched_idxs = np.where(~unmatched_mask)[0]
    if len(matched_idxs) > 0:
        registry_mod.update_matched_centroids(
            reg, Z[matched_idxs], [assigned[i] for i in matched_idxs], run_id,
        )
    registry_mod.save_registry(reg)

    labels = np.array(assigned, dtype=int)
    print(f"[cluster] {n_matched_existing} image(s) matched existing lineages, "
          f"{n_new_lineages} new lineage(s) discovered among {int(unmatched_mask.sum())} unmatched image(s)")

    # reset run-scoped image dir
    if clustered_images_dir.exists():
        shutil.rmtree(clustered_images_dir)
    clustered_images_dir.mkdir(parents=True, exist_ok=True)

    # copy images into clusters (folder name = stable lineage_id, not a per-run-arbitrary index)
    for name, lab in zip(filenames, labels):
        src = seg_dir / name
        cdir = clustered_images_dir / f"cluster_{lab}"
        cdir.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy(src, cdir / name)

    # save csv
    df = pd.DataFrame({"filename": filenames, "cluster": labels})
    df.to_csv(run_out_dir / "clusters.csv", index=False)

    # 2D viz for plot (PCA fit fresh on this run's images, just for visualization)
    Zs_viz = StandardScaler().fit_transform(Z)
    n_viz_comp = min(2, Zs_viz.shape[0] - 1, Zs_viz.shape[1])
    if n_viz_comp >= 2:
        p2 = PCA(n_components=2).fit_transform(Zs_viz)
        plt.figure(figsize=(7,6))
        plt.scatter(p2[:,0], p2[:,1], c=labels, s=18)
        plt.title(f"CLIP embeddings by lineage (run={run_id})")
        plt.tight_layout()
        plt.savefig(run_out_dir / "cluster_plot.png", dpi=160)
        plt.close()
    print(f"[cluster] Wrote clusters to {clustered_images_dir}")

    np.save(run_out_dir / "clip_embeddings.npy", Z)              # raw unit-norm CLIP
    (run_out_dir / "clip_filenames.json").write_text(json.dumps(filenames, indent=2))
    (run_out_dir / "latent_meta.json").write_text(json.dumps({
        "run_id": run_id,
        "k_new_max": k,
        "do_pca": do_pca,
        "variance": variance,
        "random_state": 0,
        "n_images": len(filenames),
        "dims_raw": int(Z.shape[1]),
        "n_matched_existing_lineages": n_matched_existing,
        "n_new_lineages": n_new_lineages,
    }, indent=2))
    print(f"[embed] Saved run artifacts to {run_out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=date.today().isoformat(),
                     help="Identifier for the scrape/segmentation run to cluster, defaults to today's date (YYYY-MM-DD)")
    ap.add_argument("--k", type=int, default=7, help="number of clusters")
    ap.add_argument("--no-pca", action="store_true", help="disable PCA")
    ap.add_argument("--var", type=float, default=0.90, help="PCA variance retain")
    args = ap.parse_args()

    seg_dir = SEGMENTED_IMAGES_DIR / args.run_id
    run_out_dir = RUNS_DIR / args.run_id
    clustered_images_dir = CLUSTERED_IMAGES_DIR / args.run_id
    run_out_dir.mkdir(parents=True, exist_ok=True)

    with track_run("cluster_pipeline", run_id=args.run_id, k=args.k) as record:
        files = load_images(seg_dir)
        Z, names = embed_clip(files)
        cluster_and_export(
            names, Z, k=args.k, run_id=args.run_id,
            seg_dir=seg_dir, run_out_dir=run_out_dir, clustered_images_dir=clustered_images_dir,
            do_pca=(not args.no_pca), variance=args.var,
        )
        record["n_images"] = len(names)

if __name__ == "__main__":
    main()


