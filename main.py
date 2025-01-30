import base64
import json
from types import NoneType
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import requests
from db.supabase_client import SupabaseMemory
from models.agent_workflows import (
    AgentRequest,
    AgentResponse,
    AgentType,
    WorkflowPlan,
    WorkflowStep,
)
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
from models.agent_dependencies import Deps, DirectorDeps
from agents.director_agent import director_agent
from agents.github_agent import github_agent
from agents.web_search_agent import web_search_agent
import os
import uuid
import httpx
import logfire

# Load environment variables
load_dotenv()

logfire.configure()
# Initialize FastAPI app
app = FastAPI()
security = HTTPBearer()

logfire.instrument_fastapi(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase_memory = SupabaseMemory()


def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> bool:
    """Verify the bearer token against environment variable."""
    expected_token = os.getenv("API_BEARER_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=500, detail="API_BEARER_TOKEN environment variable not set"
        )
    if credentials.credentials != expected_token:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return True


async def execute_workflow_step(
    client: httpx.AsyncClient, step: WorkflowStep, agents: dict, session_id: str
) -> AgentResponse:
    """Execute a single workflow step using the appropriate agent."""
    agent = agents[step.agent_type]

    try:
        result = await agent.run(
            user_prompt=f"{step.step_description}\n\nThe provided inputs to run the agent and corresponding tools are: {step.inputs}",
            deps=getattr(Deps(session_id), step.agent_type.upper(), None),
        )

        return AgentResponse(
            agent_type=step.agent_type,
            success=True,
            result=result,
        )
    except Exception as e:
        return AgentResponse(
            agent_type=step.agent_type,
            success=False,
            result=None,
            error=str(e),
        )


async def create_steps_from_plan(
    plan_result: List[Dict[str, Any]]
) -> List[WorkflowStep]:
    """Create WorkflowStep objects from the plan result and merge consecutive steps with same agent."""
    steps = []
    current_agent = None
    current_descriptions = []
    current_inputs = {}

    for step in plan_result:
        agent_type = AgentType[step["agent_type"].upper()]

        if current_agent == agent_type:
            current_descriptions.append(step.get("step_description", ""))
            current_inputs.update(step.get("inputs", {}))
        else:
            if current_descriptions:
                workflow_step = WorkflowStep(
                    step_description=" AND ".join(current_descriptions),
                    agent_type=current_agent,  # type: ignore
                    inputs=current_inputs,
                )
                steps.append(workflow_step)
            current_agent = agent_type
            current_descriptions = [step.get("step_description")]
            current_inputs = step.get("inputs", {})

    if current_descriptions:
        workflow_step = WorkflowStep(
            step_description=" AND ".join(current_descriptions),
            agent_type=current_agent,  # type: ignore
            inputs=current_inputs,
        )
        steps.append(workflow_step)

    return steps


def get_image_description(img: Dict[str, Any]) -> str:
    """Get description of image using GPT-4 Vision via OpenRouter."""
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "http://localhost:3000"),
            "X-Title": os.getenv("OPENROUTER_TITLE", "Local Development"),
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image concisely with all the details present in the image.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{img['type']};base64,{img['base64']}"
                                },
                            },
                        ],
                    }
                ],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


def process_files_to_string(files: Optional[List[Dict[str, Any]]]) -> str:
    """Convert a list of files into a formatted string with file info and image descriptions."""
    if not files:
        return ""

    file_content = "Below are the listed files provided as context. Use the appropriate tools to extract the information from the files and use it to answer the user's query:\n\n"
    for i, file in enumerate(files, 1):
        file_type = file["type"]
        file_name = file["name"]

        if file_type in ["image/jpeg", "image/png"]:
            # For images, get description using GPT-4
            img_description = "Image description: " + get_image_description(file)
            file_content += f"{i}. {file_name} (Image)\n{img_description}\n\n"
        else:
            # For CSV/Excel/Text files, just include name and type
            file_content += f"{i}. {file_name} ({file_type})\n\n"

    return file_content


@app.post("/api/mini-jarvis", response_model=AgentResponse)
async def mini_jarvis(
    request: AgentRequest, authenticated: bool = Depends(verify_token)
):
    try:
        # Initialize session and get conversation history
        logfire.info(f"Starting session for user {request.session_id}")
        is_new_session = await supabase_memory.start_session(request.session_id)
        conversation_history = (
            []
            if is_new_session
            else await supabase_memory.fetch_conversation_history(request.session_id)
        )
        # Convert conversation history to format expected by agent
        messages = []
        for msg in conversation_history:
            msg_data = msg["message"]
            msg_type = msg_data["type"]
            msg_content = msg_data["content"]

            # Process files if they exist in the message data
            if (
                msg_type == "human"
                and "data" in msg_data
                and "files" in msg_data["data"]
            ):
                files_content = process_files_to_string(msg_data["data"]["files"])
                if files_content:
                    msg_content = f"{files_content}\n\n{msg_content}"

            msg = (
                ModelRequest(parts=[UserPromptPart(content=msg_content)])
                if msg_type == "human"
                else ModelResponse(parts=[TextPart(content=msg_content)])  # type: ignore
            )
            messages.append(msg)

        # Store user's query with files if present
        message_data = {"request_id": request.request_id}
        if request.files:
            message_data["files"] = request.files  # type: ignore

        await supabase_memory.store_message(
            session_id=request.session_id,
            message_type="human",
            content=request.query,
            data=message_data,
        )

        # Initialize HTTP client for API calls
        async with httpx.AsyncClient() as client:
            if request.files:
                user_prompt = (
                    f"{process_files_to_string(request.files)}\n\n{request.query}"
                )
            else:
                user_prompt = request.query
            plan_result = await director_agent.run(
                user_prompt=user_prompt,
                message_history=messages,  # type: ignore
                deps=DirectorDeps(
                    session_id=request.session_id,
                    global_deps=Deps(session_id=request.session_id),
                ),
            )

            # Store final response
            await supabase_memory.store_message(
                session_id=request.session_id,
                message_type="ai",
                content=plan_result.data,  # type: ignore
                data={"request_id": request.request_id},
            )

            return AgentResponse(
                success=True,
                result=plan_result.data,  # type: ignore
                agent_type=AgentType.DIRECTOR,
            )

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        # Store error message
        await supabase_memory.store_message(
            session_id=request.session_id,
            message_type="ai",
            content="I apologize, but I encountered an error processing your request.",
            data={"error": str(e), "request_id": request.request_id},
        )
        return AgentResponse(
            success=False,
            result=None,
            error=str(e),
            agent_type=AgentType.DIRECTOR,
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
