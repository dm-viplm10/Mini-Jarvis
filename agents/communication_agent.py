from dataclasses import dataclass
import httpx
from config import openrouter_api_key, openrouter_base_url
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from models.agent_dependencies import CommunicationDeps
from typing import List, Dict, Any
from datetime import datetime
import os
from tools.gmail_tools import GmailTools
from tools.calendar_tools import CalendarTools


gmail_tools = GmailTools()
calendar_tools = CalendarTools()

communication_agent = Agent(
    model=OpenAIModel(
        "google/gemini-2.0-flash-exp:free",
        api_key=openrouter_api_key,
        base_url=openrouter_base_url,
    ),
    system_prompt="""You are a Communication agent specialized in handling email and calendar operations.
    You can:
    1. Read and process emails from Gmail
    2. Get and manage Google Calendar events
    3. Create and update calendar events
    
    Your responses should be clear and actionable, focusing on the specific communication task at hand.
    """,
    deps_type=CommunicationDeps,
    retries=2,
)


@communication_agent.tool
async def read_email(
    ctx: RunContext[CommunicationDeps],
    query: str | None = None,
    label: str | None = None,
) -> List[Dict[str, Any]]:
    """Read emails from Gmail matching the query or label."""
    return await gmail_tools.read_emails(ctx, query, label)


@communication_agent.tool
async def get_calendar_events(
    ctx: RunContext[CommunicationDeps],
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> List[Dict[str, Any]]:
    """Get Google Calendar events within the specified time range."""
    return await calendar_tools.get_events(ctx, start_time, end_time)


@communication_agent.tool
async def set_calendar_event(
    ctx: RunContext[CommunicationDeps],
    title: str,
    start_time: datetime,
    end_time: datetime,
    description: str | None = None,
    attendees: List[str] | None = None,
) -> Dict[str, Any]:
    """Create or update a calendar event."""
    return await calendar_tools.create_event(
        ctx, title, start_time, end_time, description, attendees
    )
