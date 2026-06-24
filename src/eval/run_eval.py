# src/eval/run_eval.py
"""
Eval harness for the RAG retriever and the tool-use agent:

  1. Retrieval accuracy: does semantic search surface the expected cluster
     in its top-k results for a query we know the right answer to?
  2. Answer quality: does the agent's final answer mention at least one of
     the expected keywords/phrases for the question?
  3. Hallucination check: an LLM judge compares the agent's final answer
     against the actual tool outputs it received, flagging any claims not
     supported by those outputs.

Writes data/eval_report.json with per-case results and aggregate metrics.
"""
from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from openai import OpenAI
from rag.retriever import TrendRetriever
import agent as agent_mod

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"
REPORT_PATH = PROJECT_ROOT / "data" / "eval_report.json"
JUDGE_MODEL = "gpt-4o-mini"

def run_retrieval_cases(cases, retriever, top_k=3):
    results = []
    for case in cases:
        hits = retriever.search(case["query"], top_k=top_k)
        hit_cluster_ids = [h["cluster_id"] for h in hits if h["run_id"] == case["run_id"]]
        passed = case["expected_cluster_id"] in hit_cluster_ids
        results.append({
            "id": case["id"],
            "query": case["query"],
            "expected_cluster_id": case["expected_cluster_id"],
            "retrieved_cluster_ids": hit_cluster_ids,
            "passed": passed,
        })
    return results

def keyword_hit(answer: str, expected_keywords_any: list) -> bool:
    answer_lower = (answer or "").lower()
    return any(kw.lower() in answer_lower for kw in expected_keywords_any)

def judge_groundedness(client, question, answer, trace):
    tool_outputs_text = json.dumps([{"tool": t["name"], "result": t["result"]} for t in trace], default=str)
    prompt = (
        "You are grading whether an AI assistant's answer is fully grounded in the tool "
        "outputs it was given, with no fabricated facts (invented cluster ids, keywords, "
        "trend statuses, or numbers that don't appear in the tool outputs).\n\n"
        f"Question: {question}\n\n"
        f"Tool outputs (JSON): {tool_outputs_text}\n\n"
        f"Assistant's answer: {answer}\n\n"
        "Respond with strict JSON: {\"grounded\": true|false, \"reason\": \"<one sentence>\"}"
    )
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

def run_agent_cases(cases, client):
    results = []
    for case in cases:
        answer, trace = agent_mod.run_agent(case["question"], return_trace=True)
        kw_passed = keyword_hit(answer, case["expected_keywords_any"])
        judgment = judge_groundedness(client, case["question"], answer, trace)
        results.append({
            "id": case["id"],
            "question": case["question"],
            "answer": answer,
            "tool_calls": [t["name"] for t in trace],
            "keyword_check_passed": kw_passed,
            "grounded": judgment.get("grounded"),
            "groundedness_reason": judgment.get("reason"),
        })
    return results

def main():
    eval_set = json.loads(EVAL_SET_PATH.read_text())
    retriever = TrendRetriever()
    client = OpenAI()

    print(f"[eval] running {len(eval_set['retrieval_cases'])} retrieval cases")
    retrieval_results = run_retrieval_cases(eval_set["retrieval_cases"], retriever)

    print(f"[eval] running {len(eval_set['agent_cases'])} agent cases (calls OpenAI + the deployed API)")
    agent_results = run_agent_cases(eval_set["agent_cases"], client)

    retrieval_accuracy = sum(r["passed"] for r in retrieval_results) / max(len(retrieval_results), 1)
    keyword_pass_rate = sum(r["keyword_check_passed"] for r in agent_results) / max(len(agent_results), 1)
    hallucination_rate = sum(1 for r in agent_results if r["grounded"] is False) / max(len(agent_results), 1)

    report = {
        "retrieval_accuracy": round(retrieval_accuracy, 3),
        "agent_keyword_pass_rate": round(keyword_pass_rate, 3),
        "agent_hallucination_rate": round(hallucination_rate, 3),
        "retrieval_cases": retrieval_results,
        "agent_cases": agent_results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"\n[eval] retrieval_accuracy:       {report['retrieval_accuracy']}")
    print(f"[eval] agent_keyword_pass_rate:  {report['agent_keyword_pass_rate']}")
    print(f"[eval] agent_hallucination_rate: {report['agent_hallucination_rate']}")
    print(f"[eval] wrote {REPORT_PATH}")

if __name__ == "__main__":
    main()
