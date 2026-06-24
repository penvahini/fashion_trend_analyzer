# src/agent.py
"""
Tool-use agent that answers natural-language fashion-trend questions by
calling the RAG retriever (semantic search over cluster labels) and the
deployed API (exact cluster/trend/similarity lookups), grounding every
answer in real pipeline output instead of letting the LLM guess.
"""
from pathlib import Path
import os
import sys
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_ENDPOINT_FILE = PROJECT_ROOT / ".api_endpoint"
MODEL = "gpt-4o-mini"

load_dotenv(PROJECT_ROOT / ".env")

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from rag.retriever import TrendRetriever  # noqa: E402

def get_api_base_url():
    env_url = os.environ.get("API_BASE_URL")
    if env_url:
        return env_url.rstrip("/")
    if API_ENDPOINT_FILE.exists():
        return API_ENDPOINT_FILE.read_text().strip().rstrip("/")
    raise RuntimeError("No API endpoint configured. Run ./scripts/deploy_api_local.sh first.")

def api_get(path: str):
    resp = requests.get(f"{get_api_base_url()}{path}", timeout=10)
    resp.raise_for_status()
    return resp.json()

_retriever = None
def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = TrendRetriever()
    return _retriever

# ---- Tool implementations -------------------------------------------------

def tool_search_trends(query: str, top_k: int = 5):
    return get_retriever().search(query, top_k=top_k)

def tool_get_cluster(run_id: str, cluster_id: str):
    return api_get(f"/clusters/{cluster_id}?run_id={run_id}")

def tool_get_trend_report():
    return api_get("/trends")

def tool_similarity_search(filename: str, run_id: str = None, top_k: int = 5):
    path = f"/similarity-search?filename={filename}&top_k={top_k}"
    if run_id:
        path += f"&run_id={run_id}"
    return api_get(path)

TOOL_IMPLS = {
    "search_trends": tool_search_trends,
    "get_cluster": tool_get_cluster,
    "get_trend_report": tool_get_trend_report,
    "similarity_search": tool_similarity_search,
}

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_trends",
            "description": "Semantic search over fashion style clusters by keywords/summary. Use for open-ended questions like 'what's trending in outerwear'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language description of the style/trend to search for"},
                    "top_k": {"type": "integer", "description": "Number of results to return", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cluster",
            "description": "Get the exact keywords, summary, and image filenames for one specific cluster id in a specific run.",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                    "cluster_id": {"type": "string"},
                },
                "required": ["run_id", "cluster_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trend_report",
            "description": "Get the full trend report: every cluster lineage classified as emerging, growing, stable, declining, or fading across all scrape runs.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "similarity_search",
            "description": "Find images visually similar to a given segmented image filename, by CLIP embedding cosine similarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "run_id": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["filename"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a fashion trend analyst assistant. Answer questions about clothing trends "
    "using the provided tools to ground your answers in real clustering/labeling data -- "
    "never invent trends, cluster ids, or keywords that didn't come from a tool result. "
    "If a tool returns no relevant data, say so plainly.\n\n"
    "Three specific failure modes to avoid:\n"
    "1. similarity_search only returns filenames and a numeric similarity score -- it does NOT "
    "tell you why images are visually similar (no color/fabric/pattern description). When asked "
    "'what makes them similar', say the similarity is based on CLIP embedding distance and that "
    "you don't have a visual description of why; do not invent details like 'color palette' or "
    "'fabric texture' that aren't in the tool output.\n"
    "2. Don't upgrade vague keywords into more specific claims than they support -- e.g. a cluster "
    "labeled 'vibrant' or 'colorful' is NOT evidence of a 'neon' trend specifically. Only state a "
    "specific term (neon, pastel, etc.) if that exact term appears in the tool output.\n"
    "3. ONLY get_trend_report returns a trend's status (emerging/growing/stable/declining/fading). "
    "search_trends does NOT -- it only returns keywords/summary/similarity score from semantic "
    "search, no status field at all. If you only called search_trends and the user asks about "
    "status, either call get_trend_report too or say you don't have status info, don't guess or "
    "invent one. When status IS available, treat it as an exact categorical label, not a word to "
    "paraphrase -- if the field says 'fading', say 'fading', not a synonym like 'declining'.\n"
    "4. When listing multiple trends (e.g. summarizing a full trend report), match each trend's "
    "status to that exact trend by name/keywords -- don't transpose statuses between trends or "
    "guess based on a general impression of the list. Re-check each trend's name against its own "
    "status field in the tool output before stating it."
)

def run_agent(user_message: str, max_turns: int = 5, verbose: bool = False, return_trace: bool = False):
    client = OpenAI()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    trace = []

    for _ in range(max_turns):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SPECS, tool_choice="auto",
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return (msg.content, trace) if return_trace else msg.content

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")
            if verbose:
                print(f"[agent] calling {name}({args})")
            try:
                result = TOOL_IMPLS[name](**args)
            except Exception as e:
                result = {"error": str(e)}
            trace.append({"name": name, "args": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    final = "Sorry, I couldn't complete this within the tool-call limit."
    return (final, trace) if return_trace else final

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What fashion trends are emerging right now?"
    print(run_agent(question, verbose=True))
