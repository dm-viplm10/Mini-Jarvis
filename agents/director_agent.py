from datetime import datetime
import json
from pydantic import BaseModel, Field

from agents.web_search_agent import web_search_agent
from agents.github_agent import github_agent
from agents.data_team_agent import product_manager_agent

from config import openrouter_api_key, openrouter_base_url
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.usage import UsageLimits
from pydantic_ai.models.openai import OpenAIModel
from typing import Any, Dict, List
from models.agent_dependencies import DirectorDeps, ProductManagerDeps
from tools.web_search_tools import crawl_parallel


director_agent = Agent(
    model=OpenAIModel(
        "openai/gpt-4o-mini",
        api_key=openrouter_api_key,
        base_url=openrouter_base_url,
    ),
    system_prompt="""You are a Director agent responsible for planning and executing responses to user queries efficiently based on the tools available to you.

Response Guidelines:
- Analyze each query to determine the minimal set of tools needed
- Identify the correct input parameters for each tool and pass them to the tool
- Only use web search when explicitly requested or when current/real-time information is absolutely necessary
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
async def scrape_explicitly_defined_webpage(
    ctx: RunContext[DirectorDeps], query: str, webpage_urls: List[str]
) -> str:
    """
    Scrape a webpage based defined in the input query defined by '@webpage_url'.
    """
    response = await crawl_parallel(ctx, webpage_urls)
    return response


@director_agent.tool
async def run_github_agent(ctx: RunContext[DirectorDeps], github_url: str) -> str:
    """
    Github agent is responsible for managing tasks related to GitHub given below:
       - Get the active pull requests from a GitHub repository
       - Get the repository details and structure
       - Get the file content of a specific file from a GitHub repository
       - Get the commit history of a specific pull request or branch in the repository
       - Get the code changes (diff) from a specific pull request
       - Set or update the description of a specific pull request
    """
    response = await github_agent.run(
        user_prompt=f"Only perform the required actions on the repository based on the inputs provided. Identify the correct inputs for each tool and pass them to the tool along with the repository URL: {github_url}",
        deps=ctx.deps.global_deps.GITHUB,
    )
    return response.data


@director_agent.tool
async def run_web_research_agent(ctx: RunContext[DirectorDeps], query: str) -> str:
    """
    Web research agent is responsible for searching the web for latest relevant information based on the user query.
    """
    response = await web_search_agent.run(
        user_prompt=query,
        deps=ctx.deps.global_deps.WEB_RESEARCH,
    )
    return response.data


@director_agent.tool
async def run_data_team_manager_agent(ctx: RunContext[DirectorDeps], query: str) -> str:
    """
    Data team manager agent is responsible for analyzing data, generating insights, and managing data workflows. This includes:
    - Analyzing customer data for patterns and trends
    - Generating statistical reports and visualizations
    - Processing and cleaning raw data files
    - Building and evaluating machine learning models
    - Creating data pipelines and ETL processes
    - Providing data-driven recommendations

    Trigger this tool to extract information from text or csv files if the name of the file is provided in the query.
    """
    response = await product_manager_agent.run(
        user_prompt=query,
        deps=ctx.deps.global_deps.PRODUCT_MANAGER,
        usage_limits=UsageLimits(request_limit=4),
    )
    return response.data


@director_agent.result_validator
async def validate_response(ctx: RunContext[None], result: Any):
    return result


#     try:
#         cleaned_result = (
#             result.replace(",\n    }", "\n    }")
#             .replace("```json", "")
#             .replace("```", "")
#         )
#         result = json.loads(cleaned_result)

#         if not isinstance(result, dict):
#             raise ModelRetry("The response must be a valid JSON object")

#         if "type" not in result:
#             raise ModelRetry("The response must include a 'type' field")

#         if result["type"] == "direct_response":
#             if "response" not in result:
#                 raise ModelRetry("Direct response must include a 'response' field")
#             return result

#         if result["type"] == "workflow":
#             if "steps" not in result:
#                 raise ModelRetry("Workflow must include a 'steps' field")

#             for step in result["steps"]:
#                 if not step.get("step_description"):
#                     raise ModelRetry("Each step must have a step description")
#                 if not step.get("agent_type"):
#                     raise ModelRetry("Each step must have an agent type")
#                 if not step.get("inputs"):
#                     raise ModelRetry("Each step must have required inputs")
#             return result

#         raise ModelRetry(
#             "Invalid response type. Must be either 'direct_response' or 'workflow'"
#         )

#     except json.JSONDecodeError:
#         raise ModelRetry(
#             """
# ALWAYS return the response only in either of the two formats. Return ONLY the JSON response - no additional text or comments:
# - direct_response: A JSON object with a "response" field containing a detailed answer to the user query if the result is an answer.
#     {
#         "type": "direct_response",
#         "response": "Your detailed answer to the user query based on your knowledge"
#     }
# - workflow: A JSON array with a "steps" field containing a list of steps to be executed if the result is a workflow.
#     {
#         "type": "workflow",
#         "steps": [
#              {
#                "step_description": "Detailed description mentioning the agent, tools, and how they should be used",
#                "agent_type": "agent_name",
#                "inputs": {
#                  "required_input_1": "value1",
#                  "required_input_2": "value2"
#                }
#              }
#            ]
#         }
# """
#         )
