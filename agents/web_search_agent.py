from __future__ import annotations as _annotations

from datetime import datetime
import logfire
from models.agent_dependencies import WebResearcherDeps
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from tools.web_search_tools import fetch_web_rag_search_result
from config import openrouter_api_key, openrouter_base_url

web_search_agent = Agent(
    model=OpenAIModel(
        "gpt-4o-mini",
        api_key=openrouter_api_key,
        base_url=openrouter_base_url,
    ),
    system_prompt=f"""You are an expert at researching the web to answer user questions. You can use the tools at your disposal to research the web and then use those results to answer the user question.  Your goal is to provide accurate, comprehensive, and well-organized answers based on retrieved web content. 
    Key Capabilities:
    1. Analyze and synthesize information from given sources
    2. Maintain source attribution and cite URLs when providing information
    3. Prioritize the most relevant and recent information
    4. Use semantic search to find the most pertinent content
    5. Structure responses in a clear, logical format

    Guidelines:
    - Always cite sources using [URL] format when referencing information
    - Synthesize information rather than quoting directly

    The current date is: {datetime.now().strftime("%Y-%m-%d")}""",
    deps_type=WebResearcherDeps,
    retries=2,
)


@web_search_agent.system_prompt
def get_dynamic_prompt(ctx: RunContext[WebResearcherDeps]) -> str:
    query = (
        ctx.deps.supabase.table("messages")
        .select("message")
        .eq("session_id", ctx.deps.session_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
        .data[0]["message"]["content"]
    )
    return f"""Based on the web search results, provide a comprehensive answer to the query: "{query}"
"""


@web_search_agent.tool
async def search_web(ctx: RunContext[WebResearcherDeps], web_query: str) -> str:
    """Search the web given a query defined to answer the user's question.

    Args:
        ctx: The context.
        web_query: The query for the web search.

    Returns:
        str: The search results as a formatted string with full content.
    """
    if ctx.deps.brave_api_key is None:
        return "This is a test web search result. Please provide a Brave API key to get real search results."

    headers = {
        "X-Subscription-Token": ctx.deps.brave_api_key,
        "Accept": "application/json",
    }

    with logfire.span("calling Brave search API", query=web_query) as span:
        r = await ctx.deps.client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={
                "q": web_query,
                "count": 5,
                "text_decorations": True,
                "search_lang": "en",
            },
            headers=headers,
        )
        r.raise_for_status()
        data = r.json()
        span.set_attribute("response", data)

    web_results = data.get("web", {}).get("results", [])

    content = await fetch_web_rag_search_result(ctx, web_results, web_query)
    return content if content else "No results found for the query."
