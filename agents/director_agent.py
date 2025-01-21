import json
from pydantic import BaseModel, Field
from config import openrouter_api_key, openrouter_base_url
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from models.agent_workflows import WorkflowPlan, WorkflowStep, AgentType
from typing import List


class Response(BaseModel):
    steps: List[WorkflowStep] = Field(
        default_factory=list,
        description="List of steps to be executed",
        examples=[
            [
                WorkflowStep(
                    step_description="Get all active pull requests for the repo using the tool get_active_pull_requests",
                    agent_type=AgentType.GITHUB,
                ),
                WorkflowStep(
                    step_description="Get repo info using the tool get_repo_info",
                    agent_type=AgentType.GITHUB,
                ),
                WorkflowStep(
                    step_description="Create a new calendar event for each active pull request using the tool set_calendar_event",
                    agent_type=AgentType.COMMUNICATION,
                ),
            ]
        ],
    )


director_agent = Agent(
    model=OpenAIModel(
        "gpt-4o-mini",
        api_key=openrouter_api_key,
        base_url=openrouter_base_url,
    ),
    system_prompt="""You are a Director agent responsible for planning workflows by breaking down user queries into actionable steps.
    You have access to the following agents and their capabilities:
    1. github:
       - get_repo_info: Get github repository information
       - get_file_content: Get content of a specific file based on the file path
       - get_repo_structure: Get repository file structure
       - get_active_pull_requests: Get active pull requests
    2. communication:
       - read_email: Read emails from Gmail
       - get_calendar_events: Get Google Calendar events
       - set_calendar_event: Create/update calendar events
    
    Your job is to:
    1. Analyze user queries and break them down into sequential steps based on the functionality of the agents available
    2. Each step should contain:
       - description of the step that mentions the agent that should handle it, the tools that should be used and details of how the tools and agents should be used to complete the step.
       - the agent that should handle it
    3. Return the workflow plan array always in a structured format like:
    [
        {
            "step_description": "Get all active pull requests for the repo using the tool get_active_pull_requests",
            "agent_type": "github",
        },
        {
            "step_description": "Create a new calendar event for each active pull request using the tool set_calendar_event",
            "agent_type": "communication",
        },
    ]
    4. Make sure that you always return only the workflow plan array in the structured format and nothing else.
    """,
    # result_type=Response,
    retries=2,
)


@director_agent.result_validator
async def validate_response(ctx: RunContext[None], result: str):
    try:
        cleaned_result = result.replace(",\n    }", "\n    }")
        result = json.loads(cleaned_result)
    except json.JSONDecodeError:
        raise ModelRetry("The response is not a valid JSON object")
    for step in result:
        if not step.get("step_description"):  # type: ignore
            raise ModelRetry("The response doesn't have step description")
        if not step.get("agent_type"):  # type: ignore
            raise ModelRetry("The response doesn't have agent type")

    return result
