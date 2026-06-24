# src/label_clusters.py
"""
Sends a handful of representative images per cluster (closest to the
cluster's CLIP centroid -- see pick_representatives) to GPT-4o-mini and asks
for keywords + a one-line trend summary. Representatives, not the full
cluster, to keep both image-upload cost and token usage bounded regardless
of cluster size. Subject to OpenAI per-minute token rate limits on a fresh
account; --clusters + --append let you retry just the clusters that 429'd
without re-labeling everything (see the retry loop in call_openai_on_images).
"""
from pathlib import Path
import os, json, argparse, base64, mimetypes, time, math
from datetime import date
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError

from observability import track_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

RUNS_DIR             = PROJECT_ROOT / "data" / "runs"
SEGMENTED_IMAGES_DIR = PROJECT_ROOT / "images" / "segmented_images"
ORIGINAL_IMAGES_DIR  = PROJECT_ROOT / "images" / "original_images"
CLUSTERED_IMAGES_DIR = PROJECT_ROOT / "images" / "clustered_images"

def load_embeddings(run_dir: Path):
    Z = np.load(run_dir / "clip_embeddings.npy")
    filenames = json.loads((run_dir / "clip_filenames.json").read_text())
    return Z, filenames

def load_clusters_df(run_dir: Path):
    return pd.read_csv(run_dir / "clusters.csv")

def to_original_filename(seg_name: str, orig_dir: Path) -> str:
    # seg_name looks like 'product_0_0_segmented.png' (one file per source
    # photo, whole outfit) -- the original stem is everything before '_segmented'.
    stem = seg_name.split("_segmented")[0]
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = orig_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate.name
    return f"{stem}.jpg"

def encode_image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def pick_representatives(Z, filenames, df, orig_dir, seg_dir, per_cluster=4, use_original=True):
    name_to_idx = {n:i for i,n in enumerate(filenames)}
    reps = {}
    for cid, sub in df.groupby("cluster"):
        names = [n for n in sub["filename"].tolist() if n in name_to_idx]
        if not names:
            reps[int(cid)] = []
            continue
        idxs = [name_to_idx[n] for n in names]
        Xc = Z[idxs]
        centroid = Xc.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(Xc - centroid, axis=1)  # ok since unit norm
        order = np.argsort(dists)
        chosen = [names[i] for i in order[:per_cluster]]

        paths = []
        for seg_name in chosen:
            if use_original:
                orig_name = to_original_filename(seg_name, orig_dir)
                op = orig_dir / orig_name
                if op.exists():
                    paths.append(op); continue
            sp = seg_dir / seg_name
            if sp.exists():
                paths.append(sp)
        reps[int(cid)] = paths
    return reps

# ---- call with retry/backoff on 429 ----
def call_openai_on_images(cluster_id, image_paths, model="gpt-4o-mini", max_retries=5):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in .env")
    client = OpenAI(api_key=api_key)

    user_content = [{
        "type": "text",
        "text": (
            "You are a fashion trend analyst. Analyze ALL images below as one cluster of clothing.\n"
            "Return a STRICT JSON object (no markdown code fences, no labels, no extra text) with EXACT shape:\n"
            "{\n"
            "  \"keywords\": [\"kw1\", \"kw2\", \"kw3\", \"kw4\", \"kw5\"],\n"
            "  \"summary\": \"1–2 sentence trend summary\"\n"
            "}\n"
            "Rules:\n"
            "- Provide 5–8 concise keywords (style, color/pattern, fabric, silhouette)\n"
            "- Do NOT include backticks or the word 'json'.\n"
        )
    }]
    for p in image_paths:
        try:
            user_content.append({"type": "image_url", "image_url": {"url": encode_image_to_data_url(p)}})
        except Exception as e:
            print(f"[warn] encode failed for {p.name}: {e}")

    attempt = 0
    while True:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Be precise, concise, and use correct fashion terminology."},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
                max_tokens=400,
                response_format={"type": "json_object"},  # ensure raw JSON object
            )
            text = (resp.choices[0].message.content or "").strip()
            data = json.loads(text)  # should be valid JSON given response_format
            break  # success
        except RateLimitError as e:
            attempt += 1
            print(f"[warn] Rate/Quota (attempt {attempt}/{max_retries}): {e}")
            if attempt >= max_retries:
                return {
                    "cluster": int(cluster_id),
                    "keywords": [],
                    "summary": "LLM quota exceeded after retries.",
                    "examples": [p.name for p in image_paths],
                }
            # exponential backoff with cap (e.g., 0.8s, 1.6s, 3.2s, 6.4s, 8.0s)
            backoff = min(0.8 * (2 ** (attempt - 1)), 8.0)
            time.sleep(backoff)
            continue
        except APIError as e:
            print(f"[warn] OpenAI API error: {e}")
            return {
                "cluster": int(cluster_id),
                "keywords": [],
                "summary": f"OpenAI error: {e}",
                "examples": [p.name for p in image_paths],
            }
        except Exception as e:
            # minimal fallback: if somehow not valid JSON, store raw text in summary
            print(f"[warn] JSON parse fallback: {e}")
            return {
                "cluster": int(cluster_id),
                "keywords": [],
                "summary": (locals().get("text") or ""),
                "examples": [p.name for p in image_paths],
            }

    # small safeguard: allow keywords to be a stringified list
    kws = data.get("keywords", [])
    if isinstance(kws, str):
        try:
            parsed = json.loads(kws)
            if isinstance(parsed, list):
                kws = parsed
            else:
                kws = [kws]
        except Exception:
            kws = [kws]
    if not isinstance(kws, list):
        kws = [str(kws)]
    kws = [str(k).strip() for k in kws if str(k).strip()]

    return {
        "cluster": int(cluster_id),
        "keywords": kws,
        "summary": str(data.get("summary", "")).strip(),
        "examples": [p.name for p in image_paths],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=date.today().isoformat(),
                     help="Identifier for the cluster run to label, defaults to today's date (YYYY-MM-DD)")
    ap.add_argument("--per-cluster", type=int, default=4)
    ap.add_argument("--use-original", action="store_true")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument(
        "--clusters", type=int, nargs="+",
        help="Only process these cluster IDs (space-separated), e.g. --clusters 5 6"
    )
    ap.add_argument(
        "--append", action="store_true",
        help="Merge results with existing cluster_labels.json by cluster id"
    )
    ap.add_argument(
        "--sleep-between", type=float, default=0.0,
        help="Optional seconds to sleep between cluster requests (helps avoid rate limits)"
    )
    ap.add_argument(
        "--max-retries", type=int, default=5,
        help="Max retries per cluster on 429 rate limits"
    )
    args = ap.parse_args()

    run_dir = RUNS_DIR / args.run_id
    orig_dir = ORIGINAL_IMAGES_DIR / args.run_id
    seg_dir = SEGMENTED_IMAGES_DIR / args.run_id
    OUT_JSON = run_dir / "cluster_labels.json"
    OUT_CSV = run_dir / "cluster_labels.csv"

    with track_run("label_clusters", run_id=args.run_id, model=args.model) as record:
        print(f"[label] load embeddings/clusters for run '{args.run_id}'")
        Z, filenames = load_embeddings(run_dir)
        df = load_clusters_df(run_dir)

        # Restrict to specific clusters if provided
        if args.clusters:
            df = df[df["cluster"].isin(args.clusters)]
            print(f"[label] restricting to clusters: {sorted(set(args.clusters))}")

        print("[label] pick representatives")
        reps = pick_representatives(Z, filenames, df, orig_dir, seg_dir, per_cluster=args.per_cluster, use_original=args.use_original)

        # If appending, load existing results
        merged_by_id = {}
        if args.append and OUT_JSON.exists():
            try:
                existing = json.loads(OUT_JSON.read_text())
                if isinstance(existing, list):
                    for x in existing:
                        if isinstance(x, dict) and "cluster" in x:
                            merged_by_id[int(x["cluster"])] = x
            except Exception as e:
                print(f"[warn] couldn't read existing {OUT_JSON}: {e}")

        results = []
        for cid in sorted(reps.keys()):
            paths = reps[cid]
            if not paths:
                continue
            print(f"[label] cluster {cid}: sending {len(paths)} images to {args.model}")
            out = call_openai_on_images(cid, paths, model=args.model, max_retries=args.max_retries)
            results.append(out)
            if args.sleep_between > 0:
                time.sleep(args.sleep_between)

        # Merge if requested
        if merged_by_id:
            for r in results:
                merged_by_id[int(r["cluster"])] = r  # overwrite or add
            results = [merged_by_id[k] for k in sorted(merged_by_id.keys())]

        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)
        record["n_clusters_labeled"] = len(results)
        print(f"[done] wrote {OUT_JSON} and {OUT_CSV}")

if __name__ == "__main__":
    main()
