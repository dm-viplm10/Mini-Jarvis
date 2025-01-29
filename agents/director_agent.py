from datetime import datetime
import json
from pydantic import BaseModel, Field
from config import openrouter_api_key, openrouter_base_url
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from models.agent_workflows import WorkflowPlan, WorkflowStep, AgentType
from typing import Any, List


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
            ]
        ],
    )


director_agent = Agent(
    model=OpenAIModel(
        "chatgpt-4o-latest",
        api_key=openrouter_api_key,
        base_url=openrouter_base_url,
    ),
    system_prompt="""You are a Director agent responsible for planning workflows by breaking down user queries into actionable steps.
    You have access to the following agents and their capabilities:
    1. github:
       - get_repo_info: Fetches detailed repository information including stars, forks, description, topics, and other metadata.
       - get_file_content: Retrieves the content of a specific file from a GitHub repository.
       - get_repo_structure: Returns the complete file and directory structure of a repository.
       - get_active_pull_requests: Fetches all open pull requests with their details like title, description, author, and review status.

    2. web_research:
       - search_web: ALWAYS use this agent to search the web to find relevant and up-to-date information for the user query based on today's date provided in the prompt.

    Your job is to:
    1. First, analyze if the user query can be handled by any of the available agents and their tools. If your knowledge cutoff is older than today's date, ALWAYSuse the web_research agent to search the web for relevant information.
    2. If NO agent can be used to handle the query:
       - Return a direct response as a JSON object in this format:
         {
           "type": "direct_response",
           "response": "Your detailed answer to the user query based on your knowledge"
         }
       - Do not attempt to create workflow steps in this case
    3. If agents CAN be used to handle the query:
       - Break down the query into sequential steps based on the functionality of available agents
       - Return a workflow plan as a JSON array in this format:
         {
           "type": "workflow",
           "steps": [
             {
               "step_description": "Detailed description mentioning the agent, tools, and how they should be used",
               "agent_type": "agent_name",
               "inputs": {
                 "required_input_1": "value1",
                 "required_input_2": "value2"
               }
             }
           ]
         }
    
    Important rules:
    1. Each step must include:
       - Clear description mentioning the agent and tools to use
       - The specific agent that should handle it
       - All required inputs (never empty)
    2. Return ONLY the JSON response - no additional text or comments
    3. All inputs must be derived from the user query or step description if not directly provided
    4. Ensure the response is always a valid JSON object
    """,
    # result_type=Response,
    retries=2,
)


@director_agent.system_prompt
def system_prompt(ctx: RunContext[None]) -> str:
    return f"""
    Today's date: {datetime.now().strftime("%Y-%m-%d")}
    """


@director_agent.result_validator
async def validate_response(ctx: RunContext[None], result: Any):
    try:
        cleaned_result = result.replace(",\n    }", "\n    }")
        result = json.loads(cleaned_result)

        if not isinstance(result, dict):
            raise ModelRetry("The response must be a valid JSON object")

        if "type" not in result:
            raise ModelRetry("The response must include a 'type' field")

        if result["type"] == "direct_response":
            if "response" not in result:
                raise ModelRetry("Direct response must include a 'response' field")
            return result

        if result["type"] == "workflow":
            if "steps" not in result:
                raise ModelRetry("Workflow must include a 'steps' field")

            for step in result["steps"]:
                if not step.get("step_description"):
                    raise ModelRetry("Each step must have a step description")
                if not step.get("agent_type"):
                    raise ModelRetry("Each step must have an agent type")
                if not step.get("inputs"):
                    raise ModelRetry("Each step must have required inputs")
            return result

        raise ModelRetry(
            "Invalid response type. Must be either 'direct_response' or 'workflow'"
        )

    except json.JSONDecodeError:
        raise ModelRetry("The response is not a valid JSON object")
