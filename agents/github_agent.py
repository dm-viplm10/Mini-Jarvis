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
    You are a GitHub specialist agent that handles repository management, PR reviews, and documentation. Analyze the user prompt carefully and perform only the required actions on the repository based on the inputs provided.
    """,
    deps_type=GithubDeps,
    retries=2,
)


@github_agent.tool
async def update_pull_request_description(
    ctx: RunContext[GithubDeps], github_url: str, pr_number: int, description: str
) -> str:
    """Update the description (body) of a specific pull request.

    Args:
        ctx: The context.
        github_url: The GitHub repository URL.
        pr_number: The pull request number.
        description: The new description/body text for the PR.

    Returns:
        str: Success or error message.
    """
    return await github_tools.update_pull_request_description(
        ctx, github_url, pr_number, description
    )


@github_agent.tool
async def get_active_prs(ctx: RunContext[GithubDeps], github_url: str) -> List[str]:
    """Get the active pull requests from a GitHub repository.

    Args:
        ctx: The context.
        github_url: The GitHub repository URL.

    Returns:
        List[str]: List of active pull requests.
    """
    return await github_tools.get_active_pull_requests(ctx, github_url)


@github_agent.tool
async def get_repo_details(ctx: RunContext[GithubDeps], github_url: str) -> str:
    """Get the details of a GitHub repository.

    Args:
        ctx: The context.
        github_url: The GitHub repository URL.

    Returns:
        str: Repository details as a formatted string.
    """
    return await github_tools.get_repo_information(ctx, github_url)


@github_agent.tool
async def get_repo_structure(ctx: RunContext[GithubDeps], github_url: str) -> str:
    """Get the directory structure of a GitHub repository.

    Args:
        ctx: The context.
        github_url: The GitHub repository URL.

    Returns:
        str: Directory structure as a formatted string.
    """
    return await github_tools.get_repo_file_structure(ctx, github_url)


@github_agent.tool
async def get_file_content(
    ctx: RunContext[GithubDeps], github_url: str, file_path: str
) -> str:
    """Get the content of a specific file from the GitHub repository.

    Args:
        ctx: The context.
        github_url: The GitHub repository URL.
        file_path: Path to the file within the repository.

    Returns:
        str: File content as a string.
    """
    return await github_tools.get_repo_file_content(ctx, github_url, file_path)


@github_agent.tool
async def get_pull_request_commits(
    ctx: RunContext[GithubDeps], github_url: str, pr_number: int
) -> str:
    """Get the commit history of a specific pull request.

    Args:
        ctx: The context.
        github_url: The GitHub repository URL.
        pr_number: The pull request number.

    Returns:
        str: Formatted string containing commit history information.
    """
    return await github_tools.get_pull_request_commits(ctx, github_url, pr_number)


@github_agent.tool
async def get_branch_commits(
    ctx: RunContext[GithubDeps], github_url: str, branch: str = "main"
) -> str:
    """Get the commit history of a specific branch in the repository.

    Args:
        ctx: The context.
        github_url: The GitHub repository URL.

    Returns:
        str: Formatted string containing branch commit history.
    """
    return await github_tools.get_branch_commits(ctx, github_url, branch)


@github_agent.tool
async def get_pull_request_changes(
    ctx: RunContext[GithubDeps], github_url: str, pr_number: int
) -> str:
    """Get the code changes (diff) from a specific pull request.

    Args:
        ctx: The context.
        github_url: The GitHub repository URL.
        pr_number: The pull request number.

    Returns:
        str: Formatted string containing the PR changes with file diffs.
    """
    return await github_tools.get_pull_request_changes(ctx, github_url, pr_number)
