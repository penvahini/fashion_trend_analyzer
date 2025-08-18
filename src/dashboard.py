# src/dashboard.py
from pathlib import Path
import json
import pandas as pd
from PIL import Image
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLUSTERS_DIR = PROJECT_ROOT / "images" / "clustered_images"
SEG_DIR      = PROJECT_ROOT / "images" / "segmented_images"
ORIG_DIR     = PROJECT_ROOT / "images" / "original_images"

st.set_page_config(page_title="Fashion Trend Clusters", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv(CLUSTERS_DIR / "clusters.csv")  # filename, cluster
    labels_path = CLUSTERS_DIR / "cluster_labels.json"
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else []
    label_by_cluster = {int(x["cluster"]): x for x in labels}
    return df, label_by_cluster

def to_original_filename(seg_name: str) -> str:
    """
    Convert a segmented filename like 'product_0_0_segmented.png'
    to the original filename 'product_0_0.jpg'. Tries common extensions.
    """
    base = seg_name
    if "_segmented" in base:
        base = base.replace("_segmented", "")
    stem = Path(base).stem
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = ORIG_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate.name
    return f"{stem}.jpg"

def show_cluster_grid(df, cid, label_by_cluster, show_both=True, ncols=4):
    sub = df[df["cluster"] == cid]
    meta = label_by_cluster.get(int(cid), {"keywords": [], "summary": "", "examples": []})

    st.subheader(f"Cluster {cid}")
    if meta["keywords"]:
        st.markdown("**Keywords:** " + ", ".join(meta["keywords"]))
    if meta["summary"]:
        st.markdown(f"**Summary:** {meta['summary']}")
    st.markdown("---")

    imgs = sub["filename"].tolist()  # segmented filenames
    if not imgs:
        st.info("No images in this cluster.")
        return

    # grid layout
    cols = st.columns(ncols, gap="small")
    for i, seg_name in enumerate(imgs):
        orig_name = to_original_filename(seg_name)
        orig_path = ORIG_DIR / orig_name
        seg_path  = SEG_DIR / seg_name

        # pick at least one that exists
        if not orig_path.exists() and not seg_path.exists():
            continue

        with cols[i % ncols]:
            if show_both and orig_path.exists() and seg_path.exists():
                st.caption(orig_name)
                c1, c2 = st.columns(2, gap="small")
                with c1:
                    st.image(Image.open(orig_path), caption="Original", use_container_width=True)
                with c2:
                    st.image(Image.open(seg_path), caption="Segmented", use_container_width=True)
            else:
                # prefer original, fallback to segmented
                final_path = orig_path if orig_path.exists() else seg_path
                st.image(Image.open(final_path), caption=final_path.name, use_container_width=True)

def main():
    st.title("Fashion Trend Analyzer — Clusters")

    st.info(
        "This dashboard shows clusters discovered from image embeddings. "
        "**Original** images are the raw inputs. **Segmented** images isolate garments to reduce background noise, "
        "improving feature extraction and clustering quality."
    )

    df, label_by_cluster = load_data()
    clusters = sorted(df["cluster"].unique().tolist())

    st.sidebar.header("Controls")
    cid = st.sidebar.selectbox("Cluster", clusters)
    show_both = st.sidebar.checkbox("Show segmented alongside original", value=True)
    ncols = st.sidebar.slider("Columns", min_value=2, max_value=6, value=4, step=1)
    st.sidebar.write("Images in cluster:", int((df["cluster"] == cid).sum()))

    show_cluster_grid(df, cid, label_by_cluster, show_both=show_both, ncols=ncols)

    st.sidebar.markdown("---")
    plot_path = CLUSTERS_DIR / "cluster_plot.png"
    if plot_path.exists():
        st.sidebar.image(str(plot_path), caption="Cluster plot", use_container_width=True)

if __name__ == "__main__":
    main()
