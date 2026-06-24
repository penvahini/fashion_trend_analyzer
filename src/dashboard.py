# src/dashboard.py
"""
Streamlit UI for the pipeline. Cluster/trend metadata is fetched from the
deployed API (api_get), not read from local files directly -- this is meant
to demonstrate "dashboard talks to the API layer," not just "dashboard reads
disk." Image *bytes* are the one exception: they're read straight from
images/original_images and images/segmented_images, since those were never
synced to S3 (see sync_artifacts_to_s3.py -- only metadata is synced, image
bytes would be a meaningfully bigger sync job for limited benefit locally).
"""
from pathlib import Path
import os
import json
import requests
import pandas as pd
import plotly.express as px
from PIL import Image
import streamlit as st

from datetime import datetime

# The dashboard is the local-dev tool (it already reads .api_endpoint, which
# only exists against LocalStack), so it's safe to default these here even
# though aws_clients.py itself doesn't -- they only apply to this process and
# anything it subprocesses (i.e. "Run New Scrape"'s sync-to-S3 step), not to
# a real deployment. Without this, launching `streamlit run src/dashboard.py`
# without first exporting these meant the sync step silently failed (it's a
# non-required step) and new runs never reached the API -- a real bug that
# was found and fixed in this session.
os.environ.setdefault("AWS_ENDPOINT_URL", "http://localhost:4566")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

import agent as agent_mod  # RAG + tool-use agent backing the Chat tab
import run_pipeline as run_pipeline_mod  # scrape -> segment -> cluster -> label -> trend orchestrator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEGMENTED_IMAGES_DIR = PROJECT_ROOT / "images" / "segmented_images"
ORIGINAL_IMAGES_DIR  = PROJECT_ROOT / "images" / "original_images"
API_ENDPOINT_FILE    = PROJECT_ROOT / ".api_endpoint"

st.set_page_config(page_title="Fashion Trend Clusters", layout="wide")

def get_api_base_url():
    env_url = os.environ.get("API_BASE_URL")
    if env_url:
        return env_url.rstrip("/")
    if API_ENDPOINT_FILE.exists():
        return API_ENDPOINT_FILE.read_text().strip().rstrip("/")
    return None

@st.cache_data(ttl=30)
def api_get(path: str):
    base_url = get_api_base_url()
    if base_url is None:
        raise RuntimeError(
            "No API endpoint configured. Run `./scripts/deploy_api_local.sh` "
            "(writes .api_endpoint) or set the API_BASE_URL env var."
        )
    resp = requests.get(f"{base_url}{path}", timeout=10)
    resp.raise_for_status()
    return resp.json()

def list_runs():
    data = api_get("/runs")
    return sorted(data["run_ids"], reverse=True)

def list_run_clusters(run_id: str):
    data = api_get(f"/runs/{run_id}/clusters")
    return data["clusters"]

def get_cluster_detail(run_id: str, cluster_id):
    return api_get(f"/clusters/{cluster_id}?run_id={run_id}")

def get_trend_report():
    return api_get("/trends")

def to_original_filename(seg_name: str, orig_dir: Path) -> str:
    """
    Convert a whole-outfit segmented filename like
    'product_0_0_segmented.png' back to the original
    'product_0_0.jpg'. Tries common extensions.
    """
    stem = seg_name.split("_segmented")[0]
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = orig_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate.name
    return f"{stem}.jpg"

def show_cluster_grid(detail: dict, orig_dir: Path, seg_dir: Path, show_both=True, ncols=4):
    st.subheader(f"Cluster {detail['cluster_id']}")
    if detail["keywords"]:
        st.markdown("**Keywords:** " + ", ".join(detail["keywords"]))
    if detail["summary"]:
        st.markdown(f"**Summary:** {detail['summary']}")
    st.markdown("---")

    imgs = detail["filenames"]  # segmented filenames
    if not imgs:
        st.info("No images in this cluster.")
        return

    # grid layout
    cols = st.columns(ncols, gap="small")
    for i, seg_name in enumerate(imgs):
        orig_name = to_original_filename(seg_name, orig_dir)
        orig_path = orig_dir / orig_name
        seg_path  = seg_dir / seg_name

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

def render_clusters_tab(run_id: str):
    orig_dir = ORIGINAL_IMAGES_DIR / run_id
    seg_dir = SEGMENTED_IMAGES_DIR / run_id

    clusters = list_run_clusters(run_id)
    if not clusters:
        st.info("No clusters found for this run.")
        return

    options = [c["cluster_id"] for c in clusters]
    labels = {c["cluster_id"]: f"Cluster {c['cluster_id']} ({c['size']} imgs)" for c in clusters}

    st.sidebar.header("Controls")
    cid = st.sidebar.selectbox("Cluster", options, format_func=lambda c: labels[c])
    show_both = st.sidebar.checkbox("Show segmented alongside original", value=True)
    ncols = st.sidebar.slider("Columns", min_value=2, max_value=6, value=4, step=1)

    detail = get_cluster_detail(run_id, cid)
    st.sidebar.write("Images in cluster:", detail["size"])

    show_cluster_grid(detail, orig_dir, seg_dir, show_both=show_both, ncols=ncols)

STATUS_ORDER = ["emerging", "growing", "stable", "declining", "fading"]
STATUS_EMOJI = {"emerging": "🌱", "growing": "📈", "stable": "➖", "declining": "📉", "fading": "🥀"}
STATUS_COLOR = {
    "emerging": "#2ecc71", "growing": "#3498db", "stable": "#95a5a6",
    "declining": "#e67e22", "fading": "#e74c3c",
}

def trend_history_df(report: dict) -> pd.DataFrame:
    rows = []
    for t in report["trends"]:
        for h in t["history"]:
            rows.append({
                "name": t["name"], "status": t["status"],
                "run_id": h["run_id"], "size": h["size"], "cluster_id": h["cluster_id"],
            })
    return pd.DataFrame(rows)

def render_trends_tab():
    try:
        report = get_trend_report()
    except Exception as e:
        st.error(f"Could not load trend report from the API: {e}")
        return

    if not report["trends"]:
        st.info("No trends found yet.")
        return

    df = trend_history_df(report)
    n_runs = len(report["run_ids"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Runs analyzed", n_runs)
    c2.metric("Tracked trend lineages", len(report["trends"]))
    c3.metric("Total images (latest run)", int(df[df["run_id"] == report["run_ids"][-1]]["size"].sum()))

    status_counts = (
        df.drop_duplicates("name")["status"].value_counts()
        .reindex(STATUS_ORDER).fillna(0).astype(int).reset_index()
    )
    status_counts.columns = ["status", "count"]

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Cluster size over time")
        if n_runs > 1:
            fig = px.line(
                df, x="run_id", y="size", color="name", markers=True,
                labels={"run_id": "Run", "size": "Cluster size (# images)", "name": "Trend"},
            )
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.4), height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.bar(
                df, x="name", y="size", color="status",
                color_discrete_map=STATUS_COLOR,
                labels={"name": "Trend", "size": "Cluster size (# images)"},
            )
            fig.update_layout(xaxis_tickangle=-30, height=420, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Only one run synced so far -- this shows cluster sizes for that run. "
                       "Add a second scrape/cluster run to see size trends over time.")

    with col_right:
        st.subheader("Trend status breakdown")
        fig2 = px.bar(
            status_counts, x="status", y="count", color="status",
            color_discrete_map=STATUS_COLOR, text="count",
        )
        fig2.update_layout(showlegend=False, height=420, xaxis_title=None, yaxis_title="# trends")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("All trends")
    latest_rows = df.sort_values("run_id").groupby("name").tail(1)
    table = latest_rows[["name", "status", "cluster_id", "run_id", "size"]].sort_values("size", ascending=False)
    table = table.rename(columns={"name": "Trend", "status": "Status", "cluster_id": "Cluster",
                                   "run_id": "Latest run", "size": "Latest size"})
    st.dataframe(table, use_container_width=True, hide_index=True)

def render_chat_tab():
    st.markdown(
        "Ask questions about the trend data. The assistant calls the same RAG + API tools "
        "exposed via MCP (`search_trends`, `get_cluster`, `get_trend_report`, `similarity_search`) "
        "-- answers are grounded in real pipeline output, not freeform generation."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("tool_calls"):
                st.caption("Tools used: " + ", ".join(turn["tool_calls"]))

    question = st.chat_input("e.g. What footwear trends are showing up?")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Calling tools and thinking..."):
                try:
                    answer, trace = agent_mod.run_agent(question, return_trace=True)
                except Exception as e:
                    answer, trace = f"Error calling the agent: {e}", []
            st.markdown(answer)
            tool_names = [t["name"] for t in trace]
            if tool_names:
                st.caption("Tools used: " + ", ".join(tool_names))

        st.session_state.chat_history.append({
            "role": "assistant", "content": answer, "tool_calls": [t["name"] for t in trace],
        })

def render_new_run_tab():
    st.markdown(
        "There's no scheduled scraper yet, so 'trends over time' only has as many data points as "
        "you've manually run. This kicks off a real new run: scrape -> segment -> cluster -> label -> "
        "trend_tracker -> sync to S3, sharing one run_id, so trend_tracker has a genuine second (or third...) "
        "data point to diff against. For real automation instead of clicking this button, see the README "
        "section on scheduling this with cron/EventBridge."
    )

    if "new_run_id_default" not in st.session_state:
        st.session_state.new_run_id_default = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    col1, col2, col3 = st.columns(3)
    with col1:
        run_id = st.text_input("Run ID", value=st.session_state.new_run_id_default)
    with col2:
        max_images = st.number_input("Max images to scrape", min_value=5, max_value=200, value=40, step=5)
    with col3:
        k = st.number_input("Number of clusters (k)", min_value=2, max_value=20, value=7, step=1)
    use_original = st.checkbox("Use original images for LLM labeling", value=True)

    st.caption(
        "Note: this runs in the same process as the dashboard, so it blocks until done (a few "
        "minutes, mostly YOLO inference). The final step syncs to LocalStack S3 (defaults to "
        "localhost:4566/test creds if not overridden) -- if LocalStack isn't running, that step "
        "will fail and you'll see a warning below; the run's local files are still saved either way."
    )

    if st.button("Start new run", type="primary"):
        log_box = st.status(f"Running pipeline for run_id={run_id}...", expanded=True)
        log_lines = []
        sync_failed = False
        try:
            for line in run_pipeline_mod.run_pipeline(
                run_id, k=int(k), max_images=int(max_images), use_original=use_original,
            ):
                if line == "[run_pipeline:STEP_FAILED] sync to S3 (for the API)":
                    sync_failed = True
                    continue  # internal sentinel, not real log output
                if line.startswith("[run_pipeline:STEP_OK]") or line.startswith("[run_pipeline:STEP_FAILED]"):
                    continue
                log_lines.append(line)
                log_box.write(line)
            log_box.update(label=f"Run {run_id} complete", state="complete")
            st.session_state.new_run_id_default = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            api_get.clear()
            if sync_failed:
                st.warning(
                    f"Run {run_id} finished locally, but syncing to S3 failed -- the API/dashboard "
                    "won't show this run until that's fixed. Make sure LocalStack is running "
                    "(`docker compose up -d localstack`) and the bucket exists "
                    "(`./scripts/setup_aws_local.sh`), then re-run `python src/sync_artifacts_to_s3.py` manually."
                )
            else:
                st.success(f"Run {run_id} finished and synced. Switch to Clusters or Trends Over Time to see it.")
        except Exception as e:
            log_box.update(label=f"Run {run_id} failed: {e}", state="error")
            st.error(str(e))

def main():
    st.title("Fashion Trend Analyzer")

    st.info(
        "This dashboard shows clusters discovered from image embeddings, and how those clusters "
        "evolve into emerging, growing, or fading trends across scrape runs. All cluster/trend data "
        "is fetched from the API Gateway + Lambda backend, not read from local files. "
        "**Original** images are the raw inputs. **Segmented** images isolate garments to reduce "
        "background noise, improving feature extraction and clustering quality."
    )

    # st.tabs resets to the first tab on any rerun (e.g. submitting the chat
    # input), since it isn't backed by session_state like other widgets are.
    # st.segmented_control with an explicit key persists the selection across
    # reruns, so submitting a chat message keeps you on the Chat view.
    view = st.segmented_control(
        "View", ["Clusters", "Trends Over Time", "Chat", "Run New Scrape"], default="Clusters", key="active_view",
    )
    if view is None:  # segmented_control allows deselecting by clicking the active option again
        view = "Clusters"

    st.markdown("---")

    if view == "Run New Scrape":
        render_new_run_tab()
        return

    if get_api_base_url() is None:
        st.error(
            "No API endpoint configured. Run `docker compose up -d localstack`, "
            "`./scripts/setup_aws_local.sh`, sync artifacts with `python src/sync_artifacts_to_s3.py`, "
            "then `./scripts/deploy_api_local.sh` (writes `.api_endpoint`)."
        )
        return

    try:
        runs = list_runs()
    except Exception as e:
        st.error(f"Could not reach the API: {e}")
        return

    if not runs:
        st.error("API reachable but no runs are synced yet. Use 'Run New Scrape' above, "
                  "or run `python src/sync_artifacts_to_s3.py` if you already have local runs.")
        return

    if view == "Clusters":
        run_id = st.selectbox("Run", runs, key="run_selector")
        render_clusters_tab(run_id)
    elif view == "Trends Over Time":
        render_trends_tab()
    elif view == "Chat":
        render_chat_tab()

if __name__ == "__main__":
    main()
