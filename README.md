# Fashion Trend Analyzer

Demo: https://drive.google.com/file/d/1ZiJyt7SKd5leM1CGajuxnkMcF_w0Saqm/view?usp=sharing

An end-to-end system that scrapes apparel images, segments outfits (YOLO + SAM), embeds them with CLIP, clusters them into style "trend lineages" that persist across scrape runs, labels trends with an LLM, and exposes everything through a serverless API, a Streamlit dashboard, and a RAG-backed chat agent (also exposed over MCP).

This project is intentionally broad rather than deep — it demonstrates a full slice of a real ML/data platform (CV, distributed ingestion, serverless APIs, RAG/agents, eval, observability) in one coherent codebase, not a production-grade SaaS. The **Future Enhancements** section below is explicit about what was cut for scope/cost and what the next step would be.

## Architecture

```
scraper --(--use-queue)--> S3 --> SQS --> ingestion_worker
   |                                            |
   v (default)                                  v
images/original_images/<run_id>/  <----- downloads from S3
   |
   v
segment.py (YOLO detect -> SAM mask, one outfit-level cutout per photo)
   |
   v
cluster_pipeline.py (CLIP embed -> match against cluster_registry.json,
   |                  KMeans only novel images)
   v
label_clusters.py (GPT-4o-mini: keywords + summary per cluster)
   |
   v
trend_tracker.py (aggregates each lineage's size per run -> 
   |               emerging/growing/stable/declining/fading)
   v
sync_artifacts_to_s3.py --> S3 --> Lambda (api_handler.py) --> API Gateway
                                       |
                       +---------------+---------------+
                       v                               v
               Streamlit dashboard          agent.py / mcp_server.py
               (Clusters/Trends/Chat/        (search_trends via RAG,
                Run New Scrape)               get_cluster, get_trend_report,
                                               similarity_search)
```

Everything left of `sync_artifacts_to_s3.py` runs locally; everything right runs on LocalStack or real AWS unchanged by flipping `AWS_ENDPOINT_URL`.

## Repository layout

```
src/
  webscraper.py              scrape one run (--use-queue for S3/SQS)
  segment.py                 YOLO + SAM outfit segmentation
  cluster_pipeline.py        CLIP embed + incremental clustering
  cluster_registry.py        persistent cross-run cluster identity
  label_clusters.py          LLM keywords/summary per cluster
  trend_tracker.py           aggregates lineages into trend report
  run_pipeline.py            orchestrates all of the above
  sync_artifacts_to_s3.py    pushes run artifacts to S3 for the API
  aws_clients.py             boto3 client factory (LocalStack or AWS)
  ingestion_worker.py        SQS consumer
  dashboard.py               Streamlit UI (API-backed)
  agent.py                   RAG + tool-use agent (OpenAI function calling)
  mcp_server.py              same tools, exposed over MCP
  observability.py / observability_report.py
  rag/                       FAISS index + retriever over cluster labels
  eval/                      eval harness (retrieval + agent groundedness)
  lambdas/                   api_handler.py, scrape_trigger_handler.py
  autoencoder_train.py       superseded approach, kept for reference
scripts/                     deploy/setup scripts (LocalStack + AWS CLI)
deployment/                  Dockerfiles + legacy run_pipeline.sh wrapper
data/runs/<run_id>/          per-run artifacts
data/cluster_registry.json   persistent lineage centroids across runs
images/{original,segmented,clustered}_images/<run_id>/
```

## Setup

Two requirements files for two jobs:

- `requirements.txt` — lean, just the Streamlit dashboard's import chain. This is what Streamlit Community Cloud installs.
- `requirements-pipeline.txt` — everything, including the heavy CV/ML stack (torch, tensorflow, ultralytics, segment-anything, playwright, transformers, open_clip, langchain).

**Running the full pipeline locally:**
```
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-pipeline.txt
playwright install chromium
```

**Running only the dashboard against an already-synced API:**
```
pip install -r requirements.txt
```

Put `OPENAI_API_KEY` in a `.env` file at the repo root (used for LLM labeling, RAG embeddings, and the agent). Model weights (`models/*.pt`, `*.pth`) are expected locally and aren't checked in.

> **Note:** A public Streamlit Cloud deploy won't show live data — `dashboard.py` reads from an API Gateway URL that only exists on a LocalStack instance running on your own machine. A public deploy shows the UI shell only; the recorded demo is the more honest way to see real functionality.

## Quick start (single machine, no AWS)

```
python src/webscraper.py
python src/segment.py
python src/cluster_pipeline.py --k 7
python src/label_clusters.py --use-original
python src/trend_tracker.py
```

Or all five in one shot:
```
python src/run_pipeline.py
```

The dashboard reads from the deployed API layer, not these local files directly (see below).

## Getting trends over time

A single run can't show a trend changing — there's nothing to compare it to. `run_pipeline.py` creates a new, comparable data point; `cluster_pipeline.py` matches each run's images against `cluster_registry.json` first, so a trend's identity stays stable across runs instead of being re-guessed.

The dashboard's **Run New Scrape** tab runs the same orchestrator with live log streaming (~10-12 min for 40 images on CPU — segmentation dominates this, see Future Enhancements). Note: scraping the same page twice within minutes won't show a real trend shift since the catalog hasn't changed — that's expected.

## Scheduling on real AWS

`run_pipeline.py` can't run inside Lambda (Playwright + YOLO + SAM are too heavy), so the real target is: scheduled trigger → small Lambda → `ecs:RunTask` against a Fargate task running the pipeline container.

```
./scripts/deploy_scrape_trigger_lambda.sh   # deploys scrape_trigger_handler.py
./scripts/schedule_new_arrivals_scrape.sh   # daily 6am UTC schedule -> Lambda
```

`scrape_trigger_handler.py` currently writes a heartbeat to S3; it'll call `ecs:RunTask` once `ECS_CLUSTER`/`ECS_TASK_DEFINITION` point at a real Fargate task built from `deployment/Dockerfile.pipeline`.

**Status:** Dockerfile written and two real build bugs fixed (missing `git` for the segment-anything dependency; missing `.dockerignore` that shipped a 2.7GB local `venv/` into the build context). Build wasn't run to completion due to a slow network link — run `docker compose --profile pipeline build pipeline` to finish verifying.

For a monthly scrape cadence, none of this is actually necessary — manually triggering a run is simpler and sufficient. This section exists so the architecture story is real, not aspirational.

**Simplest option — plain cron, no AWS:**
```
crontab -e
0 6 * * * /absolute/path/to/fashion_trend_analyzer/scripts/run_pipeline_cron.sh
```
Logs to `logs/cron/<timestamp>.log`.

## Distributed ingestion (S3 + SQS)

Simulates scrape → S3 → SQS → worker instead of saving images straight to disk. `aws_clients.py` points at LocalStack or real AWS via one env var.

```
docker compose up -d localstack
./scripts/setup_aws_local.sh
docker compose up -d --build ingestion_worker

export AWS_ENDPOINT_URL=http://localhost:4566
export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test
python src/webscraper.py --use-queue
```

## API layer (API Gateway + Lambda)

One Lambda (`api_handler.py`) behind an API Gateway REST API, reading from S3 (no filesystem access, no numpy in the Lambda — embeddings exported as plain JSON).

```
docker compose up -d localstack
./scripts/setup_aws_local.sh
export AWS_ENDPOINT_URL=http://localhost:4566
export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test
python src/trend_tracker.py
python src/sync_artifacts_to_s3.py
./scripts/deploy_api_local.sh    # prints the invoke URL

curl "<invoke_url>/trends"
curl "<invoke_url>/clusters/0"
curl "<invoke_url>/runs"
curl "<invoke_url>/runs/<run_id>/clusters"
curl "<invoke_url>/similarity-search?filename=<segmented_filename>&top_k=5"
```

Re-run `sync_artifacts_to_s3.py` after each clustering/labeling run; re-run `deploy_api_local.sh` after editing `api_handler.py`.

## Dashboard

Reads cluster/trend metadata from the API; image bytes come from local disk. `deploy_api_local.sh` writes the invoke URL to `.api_endpoint` (override with `API_BASE_URL`).

```
streamlit run src/dashboard.py
```

Four views: **Clusters** (browse by run/cluster), **Trends Over Time** (Plotly line chart + status breakdown), **Chat** (talks to `agent.py`, shows tool calls), **Run New Scrape** (triggers the pipeline with live logs).

## RAG + agent + MCP layer

A FAISS index over cluster keywords/summaries (semantic search), a tool-use agent grounded in real pipeline data, and an MCP server exposing the same tools.

```
python src/rag/build_index.py
python src/agent.py "What trends are emerging right now?"
```

Four tools: `search_trends` (RAG), `get_cluster`/`get_trend_report`/`similarity_search` (via the API). `mcp_server.py` exposes the same four tools over MCP. Re-run `build_index.py` after labeling a new run.

## Eval harness

`src/eval/eval_set.json`: 6 retrieval cases + 6 agent cases (does the final answer mention expected keywords, and is it fully grounded in the tool outputs? — caught two real hallucinations during development, fixed via a tighter system prompt).

```
python src/eval/run_eval.py
# writes data/eval_report.json: retrieval_accuracy, agent_keyword_pass_rate, agent_hallucination_rate
```

## Observability

`observability.py`'s `track_run(stage, **context)` times each stage and logs to `logs/pipeline_runs.jsonl`. Wired into segmentation, clustering, labeling, and trend tracking.

```
python src/observability_report.py --last 20
```

## Design choices worth knowing

- **Incremental clustering, not independent-per-run KMeans.** Early version re-ran KMeans each run and guessed cluster matches by centroid similarity — error-prone since boundaries drift between fits. `cluster_registry.py` now matches new images against existing lineage centroids first; only novel images get clustered fresh.
- **Outfit-level segmentation, not per-garment.** Per-garment crops left 20-80% of each crop as solid black background, which CLIP picked up as a dominant feature and collapsed clustering into black-background catch-alls. Whole-outfit segmentation fixed this.
- **CLIP over a custom autoencoder.** A from-scratch model needs far more training data than a ~40-200-image monthly scrape provides. Pretrained CLIP already has strong visual-semantic representations at this volume. (`autoencoder_train.py` kept for reference.)
- **Per-run independent PCA.** PCA is fit fresh each run for KMeans preprocessing only, since dimensionality isn't comparable across runs. Cross-run comparisons use raw CLIP embeddings instead.
- **Fashion-tuned CLIP** (Marqo/marqo-fashionCLIP via open_clip) instead of generic CLIP, to weight style-relevant axes (silhouette, fabric, garment type) over lighting/pose/background. On two real test runs, this matched fewer images to existing lineages (22/40 vs. 40/40) and found more new lineages (6 vs. 0) — a different behavior, not provably "more correct" without ground truth, but the expected direction for a domain-tuned model.
- **Trend math: share-of-catalog, with a minimum-sample floor.** Raw cluster counts are wrong once runs scrape different image counts, so `trend_tracker.py` compares `cluster_size / total_images_in_run`. A `MIN_SAMPLE_SIZE` floor (5 images) gates growing/declining verdicts to avoid reporting "+50%" off a 2→3 image swing.

## Future Enhancements & Tradeoffs

This project hasn't been connected to a real AWS account (cost reasons) — everything AWS-shaped runs against LocalStack. The first production step would be a real AWS account, least-privilege IAM roles, and unsetting `AWS_ENDPOINT_URL`; application code wouldn't need to change.

**Distributed systems:** No DLQ/retry policy on the SQS queue. The ingestion worker is a single poller, not an autoscaling fleet. Single region, no DR. Only image transfer is distributed — the actual CV compute (YOLO/SAM/CLIP) still runs as one process per run.

**AI/ML:** Cluster count (`k`) is a manual CLI argument, not data-driven (elbow method, HDBSCAN). SAM ViT-H is the slowest checkpoint (~13s/image, the dominant pipeline cost) — ViT-B/L would trade mask precision for 3-6x speed. The cluster-registry match threshold (0.75 cosine similarity) is hand-picked, not tuned against labeled data. No prompt-injection hardening on the agent (low risk today since tool outputs originate from other LLM calls, not user input). Eval set is small (12 cases) — enough to catch real bugs, not enough for strong accuracy claims at scale. Single retailer — multi-retailer scraping was scoped out due to per-site selector maintenance burden.

**Backend/API:** No auth, no rate limiting, no pagination, no server-side caching (client-side mitigated by Streamlit's 30s cache). REST was chosen over GraphQL since access patterns are a handful of fixed shapes.

**Security:** Secrets live in `.env`/env vars, not a secrets manager. IAM roles are dummy roles (LocalStack doesn't enforce them) — real least-privilege policies are a hard requirement before touching a real account. No VPC isolation, no explicit encryption-at-rest config.

**Testing/CI/CD:** No automated test suite — verification happened by running each stage against real services and inspecting results (caught a cluster-registry double-counting bug, two real agent hallucinations). Unit tests for pure logic (cluster matching, trend aggregation, Lambda routing) would be the highest-value starting point. No CI pipeline, no infrastructure-as-code (deployment is bash + AWS CLI; Terraform/CDK/SAM would be the real next step).

**Monitoring:** `pipeline_runs.jsonl` is a local log file with no alerting. A real deployment would ship these to CloudWatch with alarms on consecutive failures.
