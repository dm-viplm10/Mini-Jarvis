from datetime import datetime
import json
from pydantic import BaseModel, Field
from agents.web_search_agent import web_search_agent
from config import openrouter_api_key, openrouter_base_url
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from agents.github_agent import github_agent
from models.agent_workflows import WorkflowStep, AgentType
from models.agent_dependencies import DirectorDeps
from typing import Any, List


director_agent = Agent(
    model=OpenAIModel(
        "openai/gpt-4o-2024-05-13",
        api_key=openrouter_api_key,
        base_url=openrouter_base_url,
    ),
    system_prompt="""You are a Director agent responsible for planning and executing responses to user queries efficiently.

Response Guidelines:
- Analyze each query to determine the minimal set of tools needed
- Only use web search when explicitly requested or when current/real-time information is absolutely necessary
- Always respond in JSON format as either:
  {"type": "direct_response", "response": "message"} or 
  {"type": "tool_response", "response": "result"}
""",
    deps_type=DirectorDeps,
    # result_type=Response,
    retries=2,
)


@director_agent.system_prompt
def system_prompt(ctx: RunContext[None]) -> str:
    return f"""
    Today's date: {datetime.now().strftime("%Y-%m-%d")}
    """


@director_agent.tool
async def run_github_agent(
    ctx: RunContext[None], github_url: str, github_file_path: str
) -> str:
    """
    Github agent is responsible for managing tasks related to GitHub given below:
       - Get the active pull requests from a GitHub repository
       - Get the repository information
       - Get the file content of a specific file from a GitHub repository
       - Get the repository structure
    """
    response = await github_agent.run(
        user_prompt=f"Perform the required actions on the repository based on the tasks assigned to you. Use the below inputs to perform the actions: {github_url} and {github_file_path}",
        deps=getattr(ctx.deps.global_deps, "GITHUB", None),
    )
    return response.data


@director_agent.tool
async def run_web_research_agent(ctx: RunContext[None], query: str) -> str:
    """
    Web research agent is responsible for searching the web for latest relevant information based on the user query.
    """
    response = await web_search_agent.run(
        user_prompt=query,
        deps=getattr(ctx.deps.global_deps, "WEB_RESEARCH", None),
    )
    return response.data


@director_agent.result_validator
async def validate_response(ctx: RunContext[None], result: Any):
    try:
        cleaned_result = result.replace(",\n    }", "\n    }")
        result = json.loads(cleaned_result)

        if not isinstance(result, dict):
            raise ModelRetry("The response must be a valid JSON object")

        if "type" not in result:
            raise ModelRetry("The response must include a 'type' field")

        if result["type"] in ["direct_response", "tool_response"]:
            if "response" not in result:
                raise ModelRetry("Direct response must include a 'response' field")
            return result

        raise ModelRetry(
            "Invalid response type. Must be either 'direct_response' or 'tool_response'"
        )

    except json.JSONDecodeError:
        raise ModelRetry("The response is not a valid JSON object")
