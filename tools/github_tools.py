import re
from typing import Dict, Any, List, Tuple
import aiohttp
import os
from datetime import datetime


class GithubTool:
    def __init__(self):
        self.base_url = "https://api.github.com"

    def _parse_repo_url(self, github_url: str) -> Tuple[str, str]:
        match = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$", github_url)
        if not match:
            raise ValueError("Invalid GitHub URL format")
        owner, repo = match.groups()
        return (owner, repo)

    def _get_headers(self, github_token: str) -> Dict[str, str]:
        return {"Authorization": f"token {github_token}"} if github_token else {}

    async def get_active_pull_requests(self, ctx, github_url: str) -> List[str]:
        owner, repo = self._parse_repo_url(github_url)

        # Make request to GitHub API
        api_url = f"{self.base_url}/repos/{owner}/{repo}/pulls?state=open"
        response = await ctx.deps.client.get(
            api_url, headers=self._get_headers(ctx.deps.github_token)
        )
        if response.status != 200:
            raise Exception(f"GitHub API request failed with status {response.status}")

        pulls_data = await response.json()

        # Extract relevant PR information
        pull_requests = []
        for pr in pulls_data:
            pr_info = f"#{pr['number']} - {pr['title']} (by {pr['user']['login']})"
            pull_requests.append(pr_info)

        return pull_requests

    async def get_repo_information(self, ctx, github_url: str) -> str:
        """Get repository information including size and description using GitHub API.

        Args:
            ctx: The context.
            github_url: The GitHub repository URL.

        Returns:
            str: Repository information as a formatted string.
        """
        owner, repo = self._parse_repo_url(github_url)
        response = await ctx.deps.client.get(
            f"https://api.github.com/repos/{owner}/{repo}",
            headers=self._get_headers(ctx.deps.github_token),
        )

        if response.status_code != 200:
            return f"Failed to get repository info: {response.text}"

        data = response.json()
        size_mb = data["size"] / 1024

        return (
            f"Repository: {data['full_name']}\n"
            f"Description: {data['description']}\n"
            f"Size: {size_mb:.1f}MB\n"
            f"Stars: {data['stargazers_count']}\n"
            f"Language: {data['language']}\n"
            f"Created: {data['created_at']}\n"
            f"Last Updated: {data['updated_at']}"
        )

    async def get_repo_file_structure(self, ctx, github_url: str) -> str:
        """Get the directory structure of a GitHub repository.

        Args:
            ctx: The context.
            github_url: The GitHub repository URL.

        Returns:
            str: Directory structure as a formatted string.
        """
        owner, repo = self._parse_repo_url(github_url)
        response = await ctx.deps.client.get(
            f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1",
            headers=self._get_headers(ctx.deps.github_token),
        )

        if response.status_code != 200:
            # Try with master branch if main fails
            response = await ctx.deps.client.get(
                f"https://api.github.com/repos/{owner}/{repo}/git/trees/master?recursive=1",
                headers=self._get_headers(ctx.deps.github_token),
            )
            if response.status_code != 200:
                return f"Failed to get repository structure: {response.text}"

        data = response.json()
        tree = data["tree"]

        # Build directory structure
        structure = []
        for item in tree:
            if not any(
                excluded in item["path"]
                for excluded in [".git/", "node_modules/", "__pycache__/"]
            ):
                structure.append(
                    f"{'📁 ' if item['type'] == 'tree' else '📄 '}{item['path']}"
                )

        return "\n".join(structure)

    async def get_repo_file_content(self, ctx, github_url: str, file_path: str) -> str:
        """Get the content of a specific file from the GitHub repository.

        Args:
            ctx: The context.
            github_url: The GitHub repository URL.
            file_path: Path to the file within the repository.

        Returns:
            str: File content as a string.
        """
        owner, repo = self._parse_repo_url(github_url)
        response = await ctx.deps.client.get(
            f"https://raw.githubusercontent.com/{owner}/{repo}/main/{file_path}",
            headers=self._get_headers(ctx.deps.github_token),
        )

        if response.status_code != 200:
            # Try with master branch if main fails
            response = await ctx.deps.client.get(
                f"https://raw.githubusercontent.com/{owner}/{repo}/master/{file_path}",
                headers=self._get_headers(ctx.deps.github_token),
            )
            if response.status_code != 200:
                return f"Failed to get file content: {response.text}"

        return response.text
