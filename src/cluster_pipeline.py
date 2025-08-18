# src/cluster_pipeline.py
from pathlib import Path
import os, shutil, argparse, json
import numpy as np
import pandas as pd
import cv2
import torch
import open_clip
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEG_DIR      = PROJECT_ROOT / "images" / "segmented_images"
OUT_DIR      = PROJECT_ROOT / "images" / "clustered_images"
DATA_DIR     = PROJECT_ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224  # CLIP ViT-B/32

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
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
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

def cluster_and_export(filenames, Z, k, do_pca=True, variance=0.90, save_embeddings=False):
    Zs = StandardScaler().fit_transform(Z)
    if do_pca:
        pca_full = PCA().fit(Zs)
        n_comp = int(np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= variance)) + 1
        Zr = PCA(n_components=n_comp).fit_transform(Zs)
        print(f"[cluster] PCA -> {n_comp} comps (~{int(variance*100)}% var)")
    else:
        Zr = Zs

    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = km.fit_predict(Zr)

    # reset out dir
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # copy images into clusters
    for name, lab in zip(filenames, labels):
        src = SEG_DIR / name
        cdir = OUT_DIR / f"cluster_{lab}"
        cdir.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy(src, cdir / name)

    # save csv
    df = pd.DataFrame({"filename": filenames, "cluster": labels})
    df.to_csv(OUT_DIR / "clusters.csv", index=False)

    # 2D viz for plot
    p2 = PCA(n_components=2).fit_transform(Zr)
    plt.figure(figsize=(7,6))
    plt.scatter(p2[:,0], p2[:,1], c=labels, s=18)
    plt.title(f"CLIP + KMeans (k={k})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cluster_plot.png", dpi=160)
    plt.close()
    print(f"[cluster] Wrote clusters to {OUT_DIR}")

    if save_embeddings:
        np.save(DATA_DIR / "clip_embeddings.npy", Z)              # raw unit-norm CLIP
        (DATA_DIR / "clip_filenames.json").write_text(json.dumps(filenames, indent=2))
        np.save(DATA_DIR / "clip_embeddings_pca.npy", Zr)         # final used for KMeans
        print(f"[embed] Saved embeddings to {DATA_DIR}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=7, help="number of clusters")
    ap.add_argument("--no-pca", action="store_true", help="disable PCA")
    ap.add_argument("--var", type=float, default=0.90, help="PCA variance retain")
    ap.add_argument("--save-embeddings", action="store_true", help="dump CLIP embeddings to data/")
    args = ap.parse_args()

    files = load_images(SEG_DIR)
    Z, names = embed_clip(files)
    cluster_and_export(names, Z, k=args.k, do_pca=(not args.no_pca), variance=args.var, save_embeddings=args.save_embeddings)

if __name__ == "__main__":
    main()


