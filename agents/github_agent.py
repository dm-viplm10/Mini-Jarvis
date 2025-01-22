from config import openrouter_api_key, openrouter_base_url
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from typing import List
import os
from models.agent_dependencies import GithubDeps
from tools.github_tools import GithubTool


github_tools = GithubTool()

github_agent = Agent(
    model=OpenAIModel(
        "gpt-4o-mini",
        api_key=openrouter_api_key,
        base_url=openrouter_base_url,
    ),
    system_prompt="""
    You are a GitHub specialist agent that handles repository management, PR reviews, and documentation.
    """,
    deps_type=GithubDeps,
    retries=2,
)


@github_agent.tool
async def tool_1(ctx: RunContext[GithubDeps], github_url: str) -> List[str]:
    return await github_tools.get_active_pull_requests(ctx, github_url)


@github_agent.tool
async def tool_2(ctx: RunContext[GithubDeps], github_url: str) -> str:
    return await github_tools.get_repo_information(ctx, github_url)


@github_agent.tool
async def tool_3(ctx: RunContext[GithubDeps], github_url: str) -> str:
    return await github_tools.get_repo_file_structure(ctx, github_url)


@github_agent.tool
async def tool_4(ctx: RunContext[GithubDeps], github_url: str, file_path: str) -> str:
    return await github_tools.get_repo_file_content(ctx, github_url, file_path)
