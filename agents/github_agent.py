from dataclasses import dataclass
import httpx
from main import openrouter_api_key, openrouter_base_url
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from agents.models import WorkflowStep, AgentActions, AgentResponse
from typing import Any, Dict, List
import os
from tools.github_tools import GithubTool

@dataclass
class GithubDeps:
    client: httpx.AsyncClient
    github_token: str | None = None

github_tools = GithubTool()

github_agent = Agent(
    model=OpenAIModel(
        "google/gemini-2.0-flash-exp:free",
        api_key=openrouter_api_key,
        base_url=openrouter_base_url
    ),
    system_prompt="""
    You are a GitHub specialist agent that handles repository management, PR reviews, and documentation.
    """,
    deps_type=GithubDeps,
    retries=2
)


@github_agent.tool
async def tool_1(ctx: RunContext[GithubDeps], github_url: str) -> List[str]:
    return github_tools.get_active_pull_requests(ctx, github_url)

@github_agent.tool
async def tool_2(ctx: RunContext[GithubDeps], github_url: str) -> str:
    return github_tools.get_repo_information(ctx, github_url)

@github_agent.tool
async def tool_3(ctx: RunContext[GithubDeps], github_url: str) -> str:
    return github_tools.get_repo_file_structure(ctx, github_url)

@github_agent.tool
async def tool_4(ctx: RunContext[GithubDeps], github_url: str, file_path: str) -> str:
    return github_tools.get_repo_file_content(ctx, github_url, file_path)

