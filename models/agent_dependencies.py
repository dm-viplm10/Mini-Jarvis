import httpx
from dataclasses import dataclass
from config import github_token, brave_api_key
from db.supabase_client import SupabaseMemory, supabase
from supabase import Client


@dataclass
class GithubDeps:
    client: httpx.AsyncClient
    session_id: str
    github_token: str | None = None


@dataclass
class OrchestratorDeps:
    client: httpx.AsyncClient
    memory: SupabaseMemory
    session_id: str


@dataclass
class WebResearcherDeps:
    client: httpx.AsyncClient
    supabase: Client
    brave_api_key: str | None
    session_id: str


@dataclass
class Deps:
    def __init__(self, session_id: str):
        self.GITHUB = GithubDeps(
            client=httpx.AsyncClient(), session_id=session_id, github_token=github_token
        )
        self.ORCHESTRATOR = OrchestratorDeps(
            client=httpx.AsyncClient(), memory=SupabaseMemory(), session_id=session_id
        )
        self.WEB_RESEARCH = WebResearcherDeps(
            client=httpx.AsyncClient(),
            supabase=supabase,
            brave_api_key=brave_api_key,
            session_id=session_id,
        )


@dataclass
class DirectorDeps:
    session_id: str
    global_deps: Deps
