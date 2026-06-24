# src/lambdas/api_handler.py
"""
Single Lambda fronted by an API Gateway HTTP API. Routes:

  GET /runs
      -> list of synced run_ids (most recent last)
  GET /runs/{run_id}/clusters
      -> cluster_id, size, keywords, summary for every cluster in that run
  GET /trends
      -> full trend report (emerging/growing/stable/declining/fading clusters)
  GET /clusters/{cluster_id}?run_id=<run_id>
      -> keywords/summary/image filenames for one cluster (run_id defaults to latest)
  GET /similarity-search?filename=<name>&run_id=<run_id>&top_k=<n>
      -> nearest images to the given filename by CLIP embedding cosine similarity

Reads everything from S3 (artifacts/ prefix) -- no local filesystem, no numpy,
so the Lambda zip only needs boto3 (already in the Lambda runtime).
"""
import json
import os
import math
import boto3

S3_BUCKET = os.environ.get("S3_BUCKET", "fashion-trend-images")
AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

_s3 = None
def s3_client():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3", endpoint_url=AWS_ENDPOINT_URL, region_name=AWS_REGION)
    return _s3

def response(status, body):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }

def get_json_object(key):
    obj = s3_client().get_object(Bucket=S3_BUCKET, Key=key)
    return json.loads(obj["Body"].read())

def get_csv_rows(key):
    obj = s3_client().get_object(Bucket=S3_BUCKET, Key=key)
    text = obj["Body"].read().decode("utf-8")
    lines = text.strip().splitlines()
    header = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        values = line.split(",")
        rows.append(dict(zip(header, values)))
    return rows

def list_run_ids():
    prefix = "artifacts/"
    resp = s3_client().list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, Delimiter="/")
    run_ids = sorted(
        p["Prefix"][len(prefix):].rstrip("/")
        for p in resp.get("CommonPrefixes", [])
    )
    if not run_ids:
        raise FileNotFoundError("No synced runs found under artifacts/ in S3")
    return run_ids

def latest_run_id():
    return list_run_ids()[-1]

def handle_runs(event):
    return response(200, {"run_ids": list_run_ids()})

def handle_run_clusters(event):
    run_id = event.get("pathParameters", {}).get("run_id")
    rows = get_csv_rows(f"artifacts/{run_id}/clusters.csv")
    sizes = {}
    for r in rows:
        sizes[r["cluster"]] = sizes.get(r["cluster"], 0) + 1

    labels = get_json_object(f"artifacts/{run_id}/cluster_labels.json")
    labels_by_id = {str(l["cluster"]): l for l in labels}

    clusters = []
    for cluster_id, size in sorted(sizes.items(), key=lambda kv: int(kv[0])):
        label = labels_by_id.get(cluster_id, {})
        clusters.append({
            "cluster_id": cluster_id,
            "size": size,
            "keywords": label.get("keywords", []),
            "summary": label.get("summary", ""),
        })
    return response(200, {"run_id": run_id, "clusters": clusters})

def handle_trends(event):
    report = get_json_object("artifacts/trend_report.json")
    return response(200, report)

def handle_cluster(event):
    cluster_id = event.get("pathParameters", {}).get("cluster_id")
    query = event.get("queryStringParameters") or {}
    run_id = query.get("run_id") or latest_run_id()

    rows = get_csv_rows(f"artifacts/{run_id}/clusters.csv")
    filenames = [r["filename"] for r in rows if r["cluster"] == str(cluster_id)]

    labels = get_json_object(f"artifacts/{run_id}/cluster_labels.json")
    label = next((l for l in labels if str(l["cluster"]) == str(cluster_id)), {})

    return response(200, {
        "run_id": run_id,
        "cluster_id": cluster_id,
        "size": len(filenames),
        "filenames": filenames,
        "keywords": label.get("keywords", []),
        "summary": label.get("summary", ""),
    })

def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b + 1e-8)

def handle_similarity_search(event):
    query = event.get("queryStringParameters") or {}
    filename = query.get("filename")
    if not filename:
        return response(400, {"error": "filename query parameter is required"})
    run_id = query.get("run_id") or latest_run_id()
    top_k = int(query.get("top_k", 5))

    data = get_json_object(f"artifacts/{run_id}/clip_embeddings_export.json")
    filenames = data["filenames"]
    embeddings = data["embeddings"]

    if filename not in filenames:
        return response(404, {"error": f"filename '{filename}' not found in run '{run_id}'"})

    query_idx = filenames.index(filename)
    query_vec = embeddings[query_idx]

    sims = []
    for i, vec in enumerate(embeddings):
        if i == query_idx:
            continue
        sims.append((cosine_sim(query_vec, vec), filenames[i]))
    sims.sort(reverse=True)

    return response(200, {
        "run_id": run_id,
        "query_filename": filename,
        "results": [{"filename": f, "similarity": round(s, 4)} for s, f in sims[:top_k]],
    })

ROUTES = {
    "GET /runs": handle_runs,
    "GET /runs/{run_id}/clusters": handle_run_clusters,
    "GET /trends": handle_trends,
    "GET /clusters/{cluster_id}": handle_cluster,
    "GET /similarity-search": handle_similarity_search,
}

def lambda_handler(event, context):
    # HTTP API (v2) sends "routeKey"; REST API (v1) sends httpMethod + resource instead.
    route_key = event.get("routeKey") or f"{event.get('httpMethod')} {event.get('resource')}"
    handler = ROUTES.get(route_key)
    if handler is None:
        return response(404, {"error": f"no route for '{route_key}'"})
    try:
        return handler(event)
    except FileNotFoundError as e:
        return response(404, {"error": str(e)})
    except Exception as e:
        return response(500, {"error": str(e)})
