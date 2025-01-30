import re
from typing import Dict, Any, List, Tuple
import aiohttp
import os
from datetime import datetime


class GithubTool:
    def __init__(self):
        self.base_url = "https://api.github.com"

    def _parse_repo_url(self, github_url: str) -> Tuple[str, str]:
        # Remove any trailing slashes first
        github_url = github_url.rstrip("/")
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
        if response.status_code != 200:
            raise Exception(
                f"GitHub API request failed with status {response.status_code}"
            )

        pulls_data = response.json()

        # Extract relevant PR information
        pull_requests = []
        for pr in pulls_data:
            pr_info = f"#{pr['number']} - {pr['title']} (by {pr['user']['login']}) (URL: {pr['html_url']})"
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

    async def get_pull_request_commits(
        self, ctx, github_url: str, pr_number: int
    ) -> str:
        owner, repo = self._parse_repo_url(github_url)
        response = await ctx.deps.client.get(
            f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/commits",
            headers=self._get_headers(ctx.deps.github_token),
        )

        if response.status_code != 200:
            return f"Failed to get PR commit history: {response.text}"

        commits = response.json()
        commit_history = []

        for commit in commits:
            commit_info = {
                "sha": commit["sha"][:7],  # Short SHA
                "author": commit["commit"]["author"]["name"],
                "date": commit["commit"]["author"]["date"],
                "message": commit["commit"]["message"].split("\n")[
                    0
                ],  # First line of commit message
            }
            commit_history.append(
                f"• [{commit_info['sha']}] {commit_info['message']} "
                f"(by {commit_info['author']} on {datetime.fromisoformat(commit_info['date'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')})"
            )

        return "\n".join(
            [
                f"Commit History for PR #{pr_number}:",
                "----------------------------------------",
                *commit_history,
            ]
        )

    async def get_branch_commits(self, ctx, github_url: str, branch: str) -> str:
        owner, repo = self._parse_repo_url(github_url)
        response = await ctx.deps.client.get(
            f"{self.base_url}/repos/{owner}/{repo}/commits",
            params={"sha": branch},
            headers=self._get_headers(ctx.deps.github_token),
        )

        if response.status_code != 200:
            # If main branch fails, try master
            if branch == "main":
                response = await ctx.deps.client.get(
                    f"{self.base_url}/repos/{owner}/{repo}/commits",
                    params={"sha": "master"},
                    headers=self._get_headers(ctx.deps.github_token),
                )
                if response.status_code != 200:
                    return f"Failed to get branch commit history: {response.text}"
            else:
                return f"Failed to get branch commit history: {response.text}"

        commits = response.json()
        commit_history = []

        for commit in commits:
            commit_info = {
                "sha": commit["sha"][:7],  # Short SHA
                "author": commit["commit"]["author"]["name"],
                "date": commit["commit"]["author"]["date"],
                "message": commit["commit"]["message"].split("\n")[
                    0
                ],  # First line of commit message
                "url": commit["html_url"],
            }
            commit_history.append(
                f"• [{commit_info['sha']}] {commit_info['message']} "
                f"(by {commit_info['author']} on {datetime.fromisoformat(commit_info['date'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')}) "
                f"[View](${commit_info['url']})"
            )

        return "\n".join(
            [
                f"Commit History for branch '{branch}':",
                "----------------------------------------",
                *commit_history,
            ]
        )

    async def get_pull_request_changes(
        self, ctx, github_url: str, pr_number: int
    ) -> str:
        owner, repo = self._parse_repo_url(github_url)

        # First get the PR details to get the files changed
        response = await ctx.deps.client.get(
            f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/files",
            headers=self._get_headers(ctx.deps.github_token),
        )

        if response.status_code != 200:
            return f"Failed to get PR changes: {response.text}"

        files = response.json()
        changes_output = []

        for file in files:
            filename = file["filename"]
            status = file["status"]  # added, modified, removed
            additions = file["additions"]
            deletions = file["deletions"]
            changes = file["changes"]

            # Add file header
            changes_output.append(f"\n {filename}")
            changes_output.append(f"Status: {status.capitalize()}")
            changes_output.append(
                f"Changes: +{additions} -{deletions} ({changes} total)"
            )

            # Add the actual diff if it's not too large
            if changes <= 100:  # Limit diff size to avoid overwhelming output
                patch = file.get("patch", "")
                if patch:
                    changes_output.append("```diff")
                    changes_output.append(patch)
                    changes_output.append("```")
            else:
                changes_output.append("(Diff too large to display - view on GitHub)")

            changes_output.append("-" * 50)

        # Get PR summary
        summary = f"""Pull Request #{pr_number} Changes Summary:
Total files changed: {len(files)}
Total additions: {sum(f['additions'] for f in files)}
Total deletions: {sum(f['deletions'] for f in files)}
Total changes: {sum(f['changes'] for f in files)}
"""

        return "\n".join([summary, "=" * 50, *changes_output])

    async def update_pull_request_description(
        self, ctx, github_url: str, pr_number: int, description: str
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
        owner, repo = self._parse_repo_url(github_url)

        # Prepare the update data
        update_data = {"body": description}

        # Make PATCH request to update PR
        response = await ctx.deps.client.patch(
            f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}",
            headers=self._get_headers(ctx.deps.github_token),
            json=update_data,
        )

        if response.status_code != 200:
            return f"Failed to update PR description: {response.text}"

        data = response.json()
        return f"""Successfully updated PR #{pr_number} description.
Title: {data['title']}
Updated at: {data['updated_at']}
View PR: {data['html_url']}"""
