import httpx
from dataclasses import dataclass
from config import github_token
from db.supabase_client import SupabaseMemory


@dataclass
class GithubDeps:
    client: httpx.AsyncClient
    github_token: str | None = None


@dataclass
class CommunicationDeps:
    client: httpx.AsyncClient
    gmail_token: str | None = None
    calendar_token: str | None = None


@dataclass
class OrchestratorDeps:
    client: httpx.AsyncClient
    memory: SupabaseMemory


@dataclass
class Deps:
    GITHUB = GithubDeps(client=httpx.AsyncClient(), github_token=github_token)
    COMMUNICATION = CommunicationDeps(client=httpx.AsyncClient())
    ORCHESTRATOR = OrchestratorDeps(client=httpx.AsyncClient(), memory=SupabaseMemory())
