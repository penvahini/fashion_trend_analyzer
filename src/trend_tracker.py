# src/trend_tracker.py
"""
Tracks style clusters across scrape/cluster runs to detect emerging,
growing, fading, and stable fashion trends over time.

Cluster identity is stable by construction now: cluster_pipeline.py assigns
images to existing lineages (via src/cluster_registry.py) before ever
forming a new cluster, so the "cluster" id in every run's clusters.csv
already IS the persistent lineage_id. This module just aggregates each
lineage's size per run -- no post-hoc centroid matching/guessing needed
(that used to live here as a greedy cosine-similarity matcher; it's gone
because the thing it was approximating is now true by construction).

Date-aware by construction, not just run-ordinal. Growth/decline used to be
a raw size-delta between a lineage's last two appearances, treating "two
runs a day apart" and "two runs a year apart" identically -- a trend that
crept up slowly over a year would look the same as one that spiked in two
days. Run ids are now parsed into actual dates (see parse_run_date) and the
size delta is normalized to a rate-per-30-days before being compared against
GROWTH_THRESHOLD, so the elapsed time between runs actually matters. Runs
compared within MIN_GAP_DAYS of each other (e.g. same-day reruns) aren't
trusted for growth/decline at all -- not enough elapsed time for "trend"
to mean anything -- and fall back to "stable" with a flag explaining why.

Two more statistical fixes, both found by inspecting real output rather than
guessed in advance:
1. Growth/decline is computed on SHARE of that run's catalog
   (cluster_size / total_images_in_run), not raw image count. Different runs
   can scrape different --max-images; comparing raw counts across runs of
   different sizes is comparing two different denominators and silently
   wrong as soon as run sizes differ.
2. MIN_SAMPLE_SIZE gates growing/declining: a lineage with only 1-3 images
   produces enormous, meaningless percentage swings (2->3 images is a "50%"
   move). Below the floor, the verdict falls back to "stable" with
   insufficient_sample_size=True rather than reporting a trend with no
   statistical weight behind it. Emerging/fading aren't gated this way since
   they're presence-based, not delta-based, and a small lineage is exactly
   what "just emerged" looks like.
"""
from pathlib import Path
from datetime import datetime
import json
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from observability import track_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR     = PROJECT_ROOT / "data" / "runs"
DATA_DIR     = PROJECT_ROOT / "data"

GROWTH_THRESHOLD_PER_30D = 0.20  # +/- 20% share change *per 30 days* to call growing/declining
MIN_GAP_DAYS = 1.0                # below this, don't trust a growth/decline signal at all
MIN_SAMPLE_SIZE = 5               # below this, growth/decline isn't statistically meaningful

def list_runs():
    if not RUNS_DIR.exists():
        return []
    return sorted([p.name for p in RUNS_DIR.iterdir() if p.is_dir()])

def parse_run_date(run_id: str):
    """
    Run ids are either 'YYYY-MM-DD' (manual/cron runs) or
    'YYYY-MM-DDTHH-MM-SS' (dashboard "Run New Scrape" default). Both sort
    correctly as strings, but only this gives us an actual elapsed-time
    delta instead of a position in a list. Returns None if a run_id doesn't
    match either pattern -- callers should treat that run's gaps as unknown
    rather than guessing.
    """
    if re.match(r"^\d{4}-\d{2}-\d{2}$", run_id):
        return datetime.strptime(run_id, "%Y-%m-%d")
    if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}$", run_id):
        return datetime.strptime(run_id, "%Y-%m-%dT%H-%M-%S")
    return None

def load_run_clusters(run_id: str):
    """Returns list of {cluster_id (== lineage_id), size, keywords, summary} for a run,
    plus that run's total image count (for share-of-catalog normalization)."""
    run_dir = RUNS_DIR / run_id
    df = pd.read_csv(run_dir / "clusters.csv")
    total_in_run = len(df)

    labels_path = run_dir / "cluster_labels.json"
    labels_by_cluster = {}
    if labels_path.exists():
        for x in json.loads(labels_path.read_text()):
            labels_by_cluster[int(x["cluster"])] = x

    clusters = []
    for cid, sub in df.groupby("cluster"):
        meta = labels_by_cluster.get(int(cid), {})
        clusters.append({
            "cluster_id": int(cid),
            "size": len(sub),
            "total_in_run": total_in_run,
            "keywords": meta.get("keywords", []),
            "summary": meta.get("summary", ""),
        })
    return clusters

def build_lineages(run_ids):
    """Aggregates each stable lineage_id's history directly across runs."""
    history_by_lineage = {}
    for run_id in run_ids:
        for c in load_run_clusters(run_id):
            lineage_id = c["cluster_id"]
            share = c["size"] / c["total_in_run"] if c["total_in_run"] else 0.0
            history_by_lineage.setdefault(lineage_id, []).append({
                "run_id": run_id, "cluster_id": lineage_id, "size": c["size"],
                "total_in_run": c["total_in_run"], "share_of_catalog": round(share, 4),
                "keywords": c["keywords"], "summary": c["summary"],
            })
    return [{"history": history} for history in history_by_lineage.values()]

def classify_lineage(lineage, latest_run_id, first_run_id, run_dates):
    history = lineage["history"]
    label_kws = history[-1]["keywords"] or [f"cluster_{history[-1]['cluster_id']}"]
    name = ", ".join(label_kws[:3]) if label_kws else f"cluster_{history[-1]['cluster_id']}"

    latest_date = run_dates.get(latest_run_id)
    last_seen_date = run_dates.get(history[-1]["run_id"])
    days_since_last_seen = (
        (latest_date - last_seen_date).days
        if latest_date is not None and last_seen_date is not None else None
    )

    is_present_latest = history[-1]["run_id"] == latest_run_id
    elapsed_days = None
    rate_per_30d = None
    insufficient_time_elapsed = False
    insufficient_sample_size = False

    if not is_present_latest:
        status = "fading"
    elif len(history) == 1:
        status = "stable" if history[0]["run_id"] == first_run_id else "emerging"
    else:
        prev_entry, curr_entry = history[-2], history[-1]
        prev_share, curr_share = prev_entry["share_of_catalog"], curr_entry["share_of_catalog"]
        # Guard against a zero-share baseline (lineage briefly had 0 share due to
        # an empty run) producing a divide-by-zero / infinite delta.
        delta = (curr_share - prev_share) / max(prev_share, 1e-6)

        prev_date = run_dates.get(prev_entry["run_id"])
        curr_date = run_dates.get(curr_entry["run_id"])
        if prev_date is not None and curr_date is not None:
            elapsed_days = (curr_date - prev_date).days

        if min(prev_entry["size"], curr_entry["size"]) < MIN_SAMPLE_SIZE:
            insufficient_sample_size = True
            rate = 0.0
        elif elapsed_days is None:
            # Can't parse a real date for one of these runs -- fall back to
            # the old "raw delta between consecutive runs" behavior rather
            # than silently producing a wrong rate.
            rate = delta
        elif elapsed_days < MIN_GAP_DAYS:
            # Too little real time between these two runs (e.g. clicking
            # "Run New Scrape" twice in one day) to say anything meaningful
            # about a trend -- the catalog itself hasn't had time to change.
            insufficient_time_elapsed = True
            rate = 0.0
        else:
            rate = delta * (30.0 / elapsed_days)  # normalize to "% change per 30 days"
            rate_per_30d = round(rate, 4)

        if insufficient_time_elapsed or insufficient_sample_size:
            status = "stable"
        elif rate >= GROWTH_THRESHOLD_PER_30D:
            status = "growing"
        elif rate <= -GROWTH_THRESHOLD_PER_30D:
            status = "declining"
        else:
            status = "stable"

    return {
        "name": name,
        "status": status,
        "elapsed_days_since_prev_appearance": elapsed_days,
        "rate_per_30d": rate_per_30d,
        "insufficient_time_elapsed": insufficient_time_elapsed,
        "insufficient_sample_size": insufficient_sample_size,
        "days_since_last_seen": days_since_last_seen,
        "latest_size": history[-1]["size"],
        "latest_share_of_catalog": history[-1]["share_of_catalog"],
        "history": history,
    }

def build_report():
    run_ids = list_runs()
    if not run_ids:
        raise RuntimeError(f"No runs found in {RUNS_DIR}. Run cluster_pipeline.py first.")

    run_dates = {r: parse_run_date(r) for r in run_ids}
    unparseable = [r for r, d in run_dates.items() if d is None]
    if unparseable:
        print(f"[trend] warning: couldn't parse a date from run_id(s) {unparseable} "
              f"(expected YYYY-MM-DD or YYYY-MM-DDTHH-MM-SS) -- growth/decline rates "
              f"involving these runs fall back to a raw per-run delta, not a real rate.")

    lineages = build_lineages(run_ids)
    trends = [classify_lineage(l, run_ids[-1], run_ids[0], run_dates) for l in lineages]
    trends.sort(key=lambda t: t["latest_size"], reverse=True)

    report = {
        "run_ids": run_ids,
        "min_sample_size": MIN_SAMPLE_SIZE,
        "growth_threshold_per_30d": GROWTH_THRESHOLD_PER_30D,
        "trends": trends,
    }
    return report

def plot_timeline(report, out_path: Path, top_n=10):
    run_ids = report["run_ids"]
    run_index = {r: i for i, r in enumerate(run_ids)}

    plt.figure(figsize=(9, 6))
    for t in report["trends"][:top_n]:
        xs = [run_index[h["run_id"]] for h in t["history"]]
        ys = [h["share_of_catalog"] * 100 for h in t["history"]]
        plt.plot(xs, ys, marker="o", label=t["name"][:30])

    plt.xticks(range(len(run_ids)), run_ids, rotation=45, ha="right")
    plt.xlabel("Run")
    plt.ylabel("Share of catalog (%)")
    plt.title("Trend share of catalog over time (top clusters)")
    plt.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=10, help="number of trends to plot")
    args = ap.parse_args()

    with track_run("trend_tracker") as record:
        report = build_report()
        out_json = DATA_DIR / "trend_report.json"
        out_json.write_text(json.dumps(report, indent=2))
        print(f"[trend] wrote {out_json} ({len(report['trends'])} trend lineages across {len(report['run_ids'])} runs)")

        plot_path = DATA_DIR / "trend_timeline.png"
        plot_timeline(report, plot_path, top_n=args.top_n)
        print(f"[trend] wrote {plot_path}")

        emerging = [t for t in report["trends"] if t["status"] == "emerging"]
        fading = [t for t in report["trends"] if t["status"] == "fading"]
        growing = [t for t in report["trends"] if t["status"] == "growing"]
        declining = [t for t in report["trends"] if t["status"] == "declining"]
        print(f"[trend] emerging:  {[t['name'] for t in emerging]}")
        print(f"[trend] growing:   {[t['name'] for t in growing]}")
        print(f"[trend] declining: {[t['name'] for t in declining]}")
        print(f"[trend] fading:    {[t['name'] for t in fading]}")
        record["n_runs"] = len(report["run_ids"])
        record["n_trends"] = len(report["trends"])

if __name__ == "__main__":
    main()
