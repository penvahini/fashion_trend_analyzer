# src/mcp_server.py
"""
MCP server exposing the same trend-analysis tools used by src/agent.py, so
any MCP host (Claude Desktop, etc.) can query the fashion trend pipeline
directly: semantic search over clusters, exact cluster/trend lookups, and
CLIP-embedding similarity search.

Run with: python src/mcp_server.py
Or via the MCP CLI dev inspector: mcp dev src/mcp_server.py
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcp.server.fastmcp import FastMCP
import agent  # reuses tool_* implementations and API/RAG wiring

mcp = FastMCP("fashion-trend-analyzer")

@mcp.tool()
def search_trends(query: str, top_k: int = 5) -> list:
    """Semantic search over fashion style clusters by keywords/summary.
    Use for open-ended questions like 'what's trending in outerwear'."""
    return agent.tool_search_trends(query, top_k=top_k)

@mcp.tool()
def get_cluster(run_id: str, cluster_id: str) -> dict:
    """Get the exact keywords, summary, and image filenames for one
    specific cluster id in a specific run."""
    return agent.tool_get_cluster(run_id, cluster_id)

@mcp.tool()
def get_trend_report() -> dict:
    """Get the full trend report: every cluster lineage classified as
    emerging, growing, stable, declining, or fading across all scrape runs."""
    return agent.tool_get_trend_report()

@mcp.tool()
def similarity_search(filename: str, run_id: str = None, top_k: int = 5) -> dict:
    """Find images visually similar to a given segmented image filename,
    by CLIP embedding cosine similarity."""
    return agent.tool_similarity_search(filename, run_id=run_id, top_k=top_k)

if __name__ == "__main__":
    mcp.run()
