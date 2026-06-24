Fashion Trend Analyzer
=======================

A live demo was previously deployed at
[fashion-trend-analyzer.streamlit.app](https://fashion-trend-analyzer.streamlit.app/),
but it's stale -- it would be running the pipeline as it existed before this
README's rewrite (no run-scoping, no distributed ingestion/API/RAG/agent
layers, single global cluster set), and the link currently redirects to a
Streamlit Cloud sign-in wall rather than loading directly, which usually
means the app has gone to sleep. Treat it as not representative of the
system described below; redeploying it against the current code is on the
list, not done yet.

An end-to-end system that scrapes apparel product images, segments each
outfit out of its background (YOLO + SAM), embeds it with CLIP, clusters
outfits into style "trend lineages" that persist across scrape runs, labels
each trend with an LLM, and exposes all of it through a serverless API, a
Streamlit dashboard, and a RAG-backed chat agent (also exposed over MCP).

This project is intentionally broad rather than deep in any one area: it's
meant to demonstrate a full slice of a real ML/data platform -- computer
vision, distributed ingestion, serverless APIs, RAG/agents, eval, and
observability -- in one coherent codebase, not to be a production-grade
SaaS. The "Future Enhancements & Tradeoffs" section at the bottom is
deliberately explicit about what was cut for scope/cost reasons and what
the next real step would be for each.

Architecture at a glance
-------------------------
    scraper --(--use-queue)--> S3 (raw images) --> SQS --> ingestion_worker
       |                                                        |
       v (default, no queue)                                    v
    images/original_images/<run_id>/  <----------- downloads from S3
       |
       v
    segment.py (YOLO detect -> SAM mask, union into one outfit-level cutout per photo)
       |
       v
    cluster_pipeline.py (CLIP embed -> match against cluster_registry.json,
       |                  KMeans only the leftover/novel images)
       v
    label_clusters.py (GPT-4o-mini: keywords + summary per cluster)
       |
       v
    trend_tracker.py (aggregates each stable lineage's size per run ->
       |               emerging / growing / stable / declining / fading)
       v
    sync_artifacts_to_s3.py --> S3 --> Lambda (api_handler.py) --> API Gateway
                                                    |
                                  +-----------------+------------------+
                                  v                                    v
                          Streamlit dashboard                 agent.py / mcp_server.py
                          (Clusters / Trends /                 (search_trends via RAG,
                           Chat / Run New Scrape)               get_cluster, get_trend_report,
                                                                 similarity_search via the API)

Everything left of "sync_artifacts_to_s3.py" runs locally (or in a
container); everything right of it can run on LocalStack or real AWS
unchanged, by flipping `AWS_ENDPOINT_URL`.

Repository layout
------------------
    src/
      webscraper.py              scrape one run (--use-queue to go via S3/SQS)
      segment.py                 YOLO + SAM, one output image per source photo (whole outfit)
      cluster_pipeline.py        CLIP embed + incremental clustering
      cluster_registry.py        persistent cross-run cluster identity
      label_clusters.py          LLM keywords/summary per cluster
      trend_tracker.py           aggregates lineages into the trend report
      run_pipeline.py            orchestrates all of the above under one run_id
      sync_artifacts_to_s3.py    pushes run artifacts to S3 for the API
      aws_clients.py             boto3 client factory (LocalStack or real AWS)
      ingestion_worker.py        SQS consumer (the "distributed worker")
      dashboard.py               Streamlit UI (API-backed)
      agent.py                   RAG + tool-use agent (OpenAI function calling)
      mcp_server.py              the same tools, exposed over MCP
      observability.py / observability_report.py   run-tracking
      rag/                       FAISS index + retriever over cluster labels
      eval/                      eval harness (retrieval + agent groundedness)
      lambdas/                   api_handler.py, scrape_trigger_handler.py
      autoencoder_train.py       superseded approach, kept for reference (see below)
    scripts/                     deploy/setup scripts (LocalStack + AWS CLI)
    deployment/                  Dockerfiles + the legacy run_pipeline.sh wrapper
    data/runs/<run_id>/          per-run artifacts (embeddings, clusters, labels)
    data/cluster_registry.json   persistent lineage centroids across all runs
    images/{original,segmented,clustered}_images/<run_id>/

Setup
------
Two requirements files, for two very different jobs:

    requirements.txt           lean -- just the Streamlit dashboard's actual
                                import chain (streamlit, requests, pandas,
                                plotly, pillow, openai, faiss-cpu, numpy).
                                This is what Streamlit Community Cloud
                                installs when deploying this repo.
    requirements-pipeline.txt   everything -- adds the heavy CV/ML stack
                                (torch, tensorflow, ultralytics,
                                segment-anything, playwright, transformers,
                                open_clip, langchain) needed to actually run
                                the scrape/segment/cluster/label pipeline.

If you're running the full pipeline locally:

    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements-pipeline.txt
    playwright install chromium

If you're only running the dashboard against an already-synced API:

    pip install -r requirements.txt

Put `OPENAI_API_KEY` in a `.env` file at the repo root (used for LLM
labeling, RAG embeddings, and the agent). Model weights (`models/*.pt`,
`models/*.pth`) are expected locally; they're large and not checked in.

**A public Streamlit Cloud deploy of this dashboard will not show live
data.** `dashboard.py` fetches all cluster/trend data from a deployed API
Gateway URL (`.api_endpoint`/`API_BASE_URL`), and that API only exists on a
LocalStack instance running on your own machine -- Streamlit Cloud's
servers can't reach `localhost`. Getting a public deploy to show real data
would mean standing up the API on real AWS (see "We have not connected this
to a real AWS account" below) and pointing `API_BASE_URL` at it via
Streamlit secrets, plus setting `OPENAI_API_KEY` as a secret for the Chat
tab. Until then, a public deploy is useful for showing the UI shell, not
live functionality -- screenshots/a recorded demo of the locally-running
app are the more honest way to show this off.

Quick start (single machine, no AWS at all)
---------------------------------------------
    python src/webscraper.py
    python src/segment.py
    python src/cluster_pipeline.py --k 7
    python src/label_clusters.py --use-original
    python src/trend_tracker.py

Or all five in one shot, under one run_id:

    python src/run_pipeline.py

The dashboard needs the API layer deployed first (see below) -- it doesn't
read these local files directly, by design.

Getting more than one data point for "trends over time"
--------------------------------------------------------
A single run can't show a trend changing -- there's nothing to compare it
to. `src/run_pipeline.py` (described above) is what creates a new,
comparable data point; `cluster_pipeline.py` matches each new run's images
against `data/cluster_registry.json` before clustering anything new, so a
trend's identity is stable across runs instead of being re-guessed each time
(see "Design choices" below for why).

The dashboard's **Run New Scrape** tab runs the same orchestrator from a
button with live log streaming, so a new run doesn't require a terminal.
It blocks the dashboard process until done (~10-12 min for 40 images on
CPU -- see the timing breakdown in "Future Enhancements" below).

Both the CLI and the dashboard button are manual triggers. Scraping the same
"new arrivals" page twice within minutes won't show a real trend shift,
since the catalog itself hasn't changed -- that's expected, not a bug; real
movement needs real time between runs.

### Scheduling it for real (what you'd deploy on real AWS)

`run_pipeline.py` can't run inside Lambda (Playwright + YOLO + SAM are too
heavy/slow for Lambda's limits), so the real target is: scheduled trigger ->
small Lambda -> `ecs:RunTask` against a Fargate task running the pipeline
container.

**EventBridge Scheduler (the AWS-native cron):**

    ./scripts/deploy_scrape_trigger_lambda.sh      # deploys src/lambdas/scrape_trigger_handler.py
    ./scripts/schedule_new_arrivals_scrape.sh       # creates a daily-6am-UTC schedule -> that Lambda

`scrape_trigger_handler.py` writes a heartbeat to S3 today (verify the
wiring with `aws lambda invoke --function-name scrape-trigger out.json`),
and will call `ecs:RunTask` instead once `ECS_CLUSTER`/`ECS_TASK_DEFINITION`
env vars point at a real Fargate task definition built from
`deployment/Dockerfile.pipeline`.

`Dockerfile.pipeline` status: written; two real build bugs found and fixed
(missing `git` for the `segment-anything` git+ dependency, and a missing
`.dockerignore` that was shipping the 2.7GB local `venv/` into the build
context). The build wasn't run to completion in development due to a slow
network link in that environment -- run `docker compose --profile pipeline
build pipeline` yourself to finish verifying it.

In practice, given a monthly scrape cadence, you probably don't need any of
this: manually triggering a run (dashboard button, CLI, or the cron script
below) is simpler and sufficient. This section exists so the architecture
story is real rather than aspirational, not because day-to-day operation
needs it. Also note: LocalStack Community accepts `scheduler
create-schedule` calls but didn't reliably auto-fire them on a timer in
testing -- the Lambda itself invokes correctly when called directly, but
EventBridge's own clock trigger wasn't confirmed firing without LocalStack
Pro. Treat this as "the shape of the real AWS deployment," not as something
self-verified end-to-end locally.

**Plain cron (simplest, no AWS involved):**

    crontab -e
    0 6 * * * /absolute/path/to/fashion_trend_analyzer/scripts/run_pipeline_cron.sh

Logs each run to `logs/cron/<timestamp>.log`. Use this if you just want it
running unattended on your own machine without standing up EventBridge.

Distributed ingestion (S3 + SQS, LocalStack for local dev)
------------------------------------------------------------
Simulates scrape -> S3 -> SQS -> worker instead of saving images straight to
disk -- `src/aws_clients.py` points at LocalStack locally or real AWS by
changing one env var, no code changes either way.

    docker compose up -d localstack
    ./scripts/setup_aws_local.sh                   # creates the S3 bucket + SQS queue
    docker compose up -d --build ingestion_worker  # worker fleet stand-in (consumes the queue)

    export AWS_ENDPOINT_URL=http://localhost:4566
    export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test
    python src/webscraper.py --use-queue           # uploads to S3 + enqueues jobs

The worker downloads each job's image from S3 into
`images/original_images/<run_id>/`, so `segment.py` onward is unaffected by
whether images arrived via the queue or straight from the scraper.

API layer (API Gateway + Lambda, LocalStack for local dev)
-------------------------------------------------------------
Exposes trend/cluster data over HTTP via one Lambda (`src/lambdas/api_handler.py`)
behind an API Gateway REST API, reading everything from S3 (no filesystem
access, no numpy in the Lambda -- embeddings are exported to plain JSON).

    docker compose up -d localstack
    ./scripts/setup_aws_local.sh
    export AWS_ENDPOINT_URL=http://localhost:4566
    export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test
    python src/trend_tracker.py              # writes data/trend_report.json
    python src/sync_artifacts_to_s3.py       # uploads clusters/labels/embeddings/trend report to S3
    ./scripts/deploy_api_local.sh            # creates the Lambda + REST API, prints the invoke URL

    curl "<invoke_url>/trends"
    curl "<invoke_url>/clusters/0"
    curl "<invoke_url>/runs"
    curl "<invoke_url>/runs/<run_id>/clusters"
    curl "<invoke_url>/similarity-search?filename=<segmented_filename>&top_k=5"

Re-run `sync_artifacts_to_s3.py` after each new clustering/labeling run so
the API serves current data; re-run `deploy_api_local.sh` after editing
`api_handler.py`.

Dashboard
----------
Reads cluster/trend metadata from the API (not local files); image *bytes*
still come from local disk since those weren't synced to S3.
`deploy_api_local.sh` writes the invoke URL to `.api_endpoint`, which
`dashboard.py` reads by default (override with `API_BASE_URL`).

    streamlit run src/dashboard.py

Four views: **Clusters** (browse images per run/cluster), **Trends Over
Time** (Plotly: size-over-time line chart once 2+ runs exist, status
breakdown, sortable table), **Chat** (talks to `src/agent.py` directly,
shows which tools it called), **Run New Scrape** (triggers `run_pipeline.py`
from a button with live logs).

RAG + agent + MCP layer
---------------------------
A FAISS index over every cluster's keywords/summary (semantic search, not
just exact cluster-id lookup), a tool-use agent that grounds answers in real
pipeline data, and an MCP server exposing the same tools to any MCP host.

    python src/rag/build_index.py          # embeds cluster labels -> data/rag_index/
    python src/agent.py "What trends are emerging right now?"

Four tools: `search_trends` (RAG), `get_cluster`/`get_trend_report`/
`similarity_search` (all via the API). `src/mcp_server.py` exposes the same
four tools over MCP (`python src/mcp_server.py`, or `mcp dev src/mcp_server.py`
to inspect interactively). Re-run `build_index.py` after labeling a new run.

Eval harness
-------------
`src/eval/eval_set.json`: 6 retrieval cases (does semantic search surface
the right cluster for a known query) + 6 agent cases (does the final answer
mention expected keywords, and does an LLM judge confirm it's fully
supported by the tool outputs the agent actually received -- this caught
two real hallucinations during development: the agent inventing visual
descriptions that `similarity_search` doesn't provide, and overclaiming
"neon" from a cluster merely labeled "vibrant"; both fixed via a tighter
system prompt in `agent.py`).

    python src/eval/run_eval.py
    # writes data/eval_report.json: retrieval_accuracy, agent_keyword_pass_rate, agent_hallucination_rate

Add cases as the pipeline grows so the eval set doesn't go stale.

Observability
---------------
`src/observability.py`'s `track_run(stage, **context)` context manager
times a stage and appends a record (status, duration, context, full
traceback on error) to `logs/pipeline_runs.jsonl`. Wired into `segment.py`,
`cluster_pipeline.py`, `label_clusters.py`, `trend_tracker.py`.

    python src/observability_report.py --last 20

Design choices worth knowing about
------------------------------------
**Incremental clustering instead of independent-per-run KMeans.** Early
version re-ran KMeans from scratch every run and then *guessed* which
cluster in run N matched which cluster in run N-1 by centroid similarity --
a real source of error, since cluster boundaries can drift between
independent fits even for the same underlying trend. The current version
(`cluster_registry.py`) makes identity stable by construction: new images
are matched against existing lineage centroids first; only genuinely novel
images get KMeans'd into new lineages. `trend_tracker.py` shrank
considerably as a result -- it just aggregates by a stable lineage_id now,
no more cross-run matching heuristic.

**Outfit-level segmentation, not per-garment.** We tried per-garment
segmentation (cropping+masking each YOLO detection separately, tagged with
its class) to make "what's trending in tops" vs. "what's trending in shoes"
independently answerable. In practice it made clustering meaningfully
*worse*: a per-garment crop's bounding box is small relative to its mask, so
20-80% of each crop ended up solid black background (measured directly).
CLIP picked up "this image is mostly a uniform black region" as a dominant
feature shared across totally unrelated garments, collapsing clustering
into a couple of black-background catch-alls instead of real style
groupings. Reverted to the original approach: union all of an image's YOLO
detections into one mask, one cutout per source photo (whole outfit). The
black-background artifact is much smaller relative to the whole frame at
outfit scale, so it stopped dominating the embedding. The real lesson:
"finer-grained" isn't automatically better for an embedding model that's
sensitive to dominant visual features you didn't intend to introduce --
per-garment trend granularity would need a different fix (e.g. embedding on
a neutral/inpainted background instead of solid black, or not masking the
background at all and relying on a tight bbox crop with real context) if
it's revisited later.

**CLIP over a custom autoencoder.** `src/autoencoder_train.py` is an earlier
experiment training a from-scratch convolutional autoencoder for
embeddings; it's not in the active pipeline. A from-scratch model needs far
more training images per run than a ~40-200-image monthly scrape provides;
a pretrained CLIP model already has strong general visual-semantic
representations and works well at this data volume. Kept for reference, not
deleted, since it's a legitimate design comparison worth being able to talk
through.

**Per-run independent PCA, not shared across runs.** `cluster_pipeline.py`
fits PCA fresh each run (for KMeans preprocessing only) since PCA
dimensionality depends on that run's data and isn't comparable across runs.
All cross-run comparisons (registry matching, trend tracking) use raw CLIP
embeddings instead, which are fixed-dimension and consistent run to run.

**Fashion-tuned CLIP instead of generic CLIP.** `cluster_pipeline.py` now
embeds with `Marqo/marqo-fashionCLIP` (loaded via `open_clip`'s hf-hub
support) instead of generic `openai/CLIP ViT-B-32`. Generic CLIP was trained
on everything (people, objects, scenes) so it's sensitive to lighting/pose/
background as well as garment style; a fashion-tuned model should weight
genuinely style-relevant axes (silhouette, fabric, garment type) more
heavily. Observed effect on the same two real runs: the fashion-tuned model
matched far fewer images to existing lineages (22/40 vs. 40/40 with generic
CLIP) and discovered more new lineages (6 vs. 0) -- it draws finer-grained
distinctions. That's a different clustering behavior, not provably "more
correct" without labeled ground truth, but it's the expected direction for
a domain-tuned embedding model. Note: `transformers`' `CLIPModel`/
`CLIPProcessor` classes crash in this dev environment specifically (a
reproducible bus error in the vision-tower forward pass, and a separate
mutex crash in the image processor -- both checkpoint/environment-specific,
reproduced even with the generic `openai/clip-vit-base-patch32` checkpoint,
so not a code bug here) -- `open_clip`'s loading path avoids both issues
entirely and needs no preprocessing changes, since this checkpoint uses the
same 224x224 size and CLIP normalization constants as the model it replaces.

**Trend math: share-of-catalog, not raw count, plus a minimum-sample
floor.** Two real distortions, found by inspecting actual output rather
than guessed in advance: (1) comparing raw cluster image counts across runs
is wrong as soon as runs scrape different `--max-images` -- `trend_tracker.py`
now compares `cluster_size / total_images_in_run` (share of that run's
catalog) instead. (2) Lineages with only 1-3 images produced enormous,
meaningless percentage swings (2->3 images reads as "+50%"). A
`MIN_SAMPLE_SIZE` floor (5 images) now gates growing/declining specifically
-- below it, the verdict falls back to "stable" with
`insufficient_sample_size: true` rather than reporting a trend with no
real statistical weight behind it. Emerging/fading aren't gated this way
since they're presence-based, not delta-based -- a small lineage is exactly
what "just emerged" looks like.

Future Enhancements & Tradeoffs
===================================
This section is deliberately explicit about what's cut for scope/cost
reasons, what the tradeoff was, and what the real next step would be --
the kind of thing worth being able to discuss in an interview rather than
silently glossing over.

**We have not connected this to a real AWS account, for cost reasons.**
Everything AWS-shaped in this repo (S3, SQS, Lambda, API Gateway, IAM,
EventBridge) runs against LocalStack, which is free and behaviorally close
to real AWS for the services used here, but isn't real AWS. The first step
in actually taking this to production would be standing up a real AWS
account, real IAM roles with least-privilege policies (the current IAM
roles are dummy roles LocalStack doesn't enforce), and pointing
`AWS_ENDPOINT_URL` at nothing (i.e. unsetting it) -- the application code
doesn't need to change, only the deployment scripts' target.

Distributed systems
---------------------
- **Queue choice (SQS vs. alternatives).** SQS was chosen for simplicity --
  no broker to run, natural fit for "one worker pulls one job." A system
  with higher throughput or multiple consumer types per event (e.g. one
  consumer for ingestion, another for analytics, another for alerting)
  would outgrow SQS's one-queue-one-consumer-group model and want SNS
  fan-out to multiple SQS queues, or Kafka/Kinesis for replayable streams.
  Not needed at this volume (tens of images per run), but the first thing
  to revisit if ingestion volume or consumer count grew.
- **No dead-letter queue or retry/backoff policy on the ingestion SQS
  queue.** A poison message (e.g. a corrupted image) would be retried
  indefinitely rather than parked for inspection. A production version
  needs a DLQ with a maxReceiveCount and alerting on DLQ depth.
- **The "distributed worker" is one long-running poller, not an
  autoscaling fleet.** `ingestion_worker.py` demonstrates the
  queue-consumer pattern but doesn't demonstrate horizontal scaling. A real
  version would run as an ECS service (or Lambda triggered by SQS directly)
  with scaling tied to queue depth.
- **Single region, no DR story.** Fine for a portfolio project; a real
  deployment would want at minimum cross-region S3 replication for the
  artifacts that took real compute/LLM-cost to produce.
- **Heavy compute (segmentation) only partially distributed.** The SQS
  worker distributes *image transfer* (S3 -> local disk), not the actual
  YOLO/SAM/CLIP compute -- that still runs as a single local/container
  process per run. True distribution would have the worker itself run
  segmentation per-message (one Fargate task or Lambda-with-EFS per image
  or per batch), parallelizing the genuinely expensive part.

AI / ML
--------
- **Fixed `k` for new-cluster discovery, not data-driven.** The number of
  *new* clusters formed each run from unmatched images is a CLI argument,
  not chosen via elbow method or silhouette score. Auto-selecting k (or
  switching to a method that doesn't require pre-specifying it, like
  HDBSCAN) would remove a manual tuning knob.
- **SAM ViT-H is the largest, slowest SAM checkpoint.** ~13s/image on CPU,
  the dominant cost in the entire pipeline (a 40-image run spends ~9 of its
  ~10-12 total minutes in segmentation). ViT-B or ViT-L trade some mask
  precision for 3-6x speed -- worth it if iteration speed ever matters more
  than mask quality.
- **Match threshold for cluster-registry identity (0.75 cosine similarity)
  is a hand-picked constant**, not tuned against labeled ground truth. Too
  high and real trend continuity gets misclassified as "new"; too low and
  unrelated images get folded into an existing trend. Worth a proper
  threshold sweep against human-labeled "same trend / different trend"
  pairs if this needs to be trustworthy at larger scale.
- **No prompt-injection hardening on the agent.** `agent.py`'s tools return
  data the LLM treats as trusted (cluster labels, summaries) -- all of
  which originated from *other LLM calls* (label_clusters.py), not
  untrusted user input, so the practical risk is low today. It would matter
  more if a future version ingested arbitrary user-supplied text (e.g. a
  product review) into a tool result.
- **Eval set is small (12 hand-written cases).** Enough to catch the two
  real hallucination bugs found during development, not enough to make
  strong claims about retrieval accuracy at scale. A real eval suite would
  need dozens-to-hundreds of cases and ideally some sourced from real user
  queries rather than written by the same person who built the system.
- **Single retailer.** Trend claims from one store's "new arrivals" page
  are a thin signal for "fashion trends" broadly. Multi-retailer scraping
  was explicitly scoped out of this round of work -- the real cost isn't
  the scraping itself, it's the ongoing maintenance burden of per-site CSS
  selectors breaking whenever a retailer redesigns their site, plus
  normalizing inconsistent category/size taxonomies across retailers.

Backend / API
---------------
- **No auth on the API.** Every endpoint in `api_handler.py` is publicly
  invokable once deployed to real AWS (LocalStack doesn't enforce IAM, so
  this wasn't visible locally). A real deployment needs at minimum an API
  key or Cognito authorizer on API Gateway, and resource-level IAM policies
  if other AWS services call it directly.
- **No rate limiting / usage plans.** API Gateway supports usage plans and
  throttling natively; none are configured. Cheap to add, just not done yet.
- **REST, not GraphQL.** REST fit the access patterns here (a handful of
  fixed shapes: trends, one cluster, similarity search) -- GraphQL's
  flexible querying would be over-engineering for this surface area, but
  would be worth reconsidering if the dashboard needed to compose many
  small queries into one request.
- **No pagination.** `/runs/{run_id}/clusters` and `/trends` return
  everything in one response. Fine at the current data volume (single-digit
  to low-double-digit clusters/lineages); would need cursor-based
  pagination if a deployment accumulated years of monthly runs.
- **No caching layer.** Every dashboard interaction re-hits the Lambda
  (mitigated client-side by Streamlit's 30s `@st.cache_data` TTL, but
  there's no CDN/API Gateway caching on the server side).

Security
---------
- **Secrets in `.env` / plain environment variables**, not a secrets
  manager. `OPENAI_API_KEY` and AWS credentials are read from `.env` or
  shell env vars. A real deployment should use AWS Secrets Manager or SSM
  Parameter Store, with the Lambda/ECS task role granted scoped read access
  -- not a credential sitting in a file.
- **Dummy IAM roles.** Every `iam create-role` call in the deploy scripts
  creates a role with a trust policy but no attached permission policy --
  LocalStack doesn't enforce IAM so this "works," but real AWS would deny
  every action. Real least-privilege policies (S3 read/write scoped to the
  specific bucket/prefix, no wildcard `*` actions) are a hard requirement
  before this touches a real account.
- **No VPC / network isolation.** Lambdas and the API are deployed with
  default networking. A production version handling anything sensitive
  would want the Lambda in a private subnet with a VPC endpoint to S3,
  rather than public internet egress.
- **No encryption-at-rest configuration specified.** S3 default encryption,
  SQS server-side encryption, and CloudWatch Logs encryption are all
  unset (relying on AWS defaults rather than an explicit KMS policy).
- **No dependency/vulnerability scanning in CI** (see below -- there's no
  CI at all yet).

Testing / CI/CD
-----------------
- **No automated test suite.** Verification in this project happened by
  actually running each stage against real services (LocalStack, real
  OpenAI calls, a real Playwright scrape) and inspecting results -- which
  caught real bugs (a cluster-registry double-counting bug, a stale
  `.dockerignore`, two real agent hallucinations) but isn't a substitute
  for a pytest suite that runs on every change. Unit tests for the pure
  logic (cluster_registry matching/centroid math, trend_tracker
  aggregation, the Lambda handler's routing) would be the highest-value
  starting point since they don't require live services.
- **No CI pipeline.** No GitHub Actions (or equivalent) running tests,
  linting, or `terraform plan`-style validation on PRs. Worth adding even
  a minimal lint+syntax-check workflow before this is anyone's "real"
  deployment target.
- **No infrastructure-as-code.** Deployment is bash scripts calling the AWS
  CLI directly (`scripts/deploy_api_local.sh`, etc.) -- fine for iterating
  against LocalStack, but a real deployment should use Terraform/CDK/SAM so
  infrastructure changes are reviewable, repeatable, and not dependent on
  someone running the right script in the right order by hand.

Monitoring / alerting
------------------------
- **`logs/pipeline_runs.jsonl` is a local file, not a metrics system.** It's
  genuinely useful for "what happened in the last N runs" but there's no
  alerting on it (a failed run just sits in the log until someone reads
  it). A real deployment would ship these events to CloudWatch Metrics/Logs
  (or a hosted alternative) with an alarm on consecutive failures.
- **No dashboards/alarms on the deployed API or Lambdas** (invocation
  errors, latency, throttling) -- CloudWatch was enabled in LocalStack
  for this project but only to make Lambda logging work, not for actual
  monitoring.
