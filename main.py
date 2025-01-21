import json
from types import NoneType
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from db.supabase_client import SupabaseMemory
from models.agent_workflows import (
    AgentRequest,
    AgentResponse,
    AgentType,
    WorkflowPlan,
    WorkflowStep,
)
from models.agent_dependencies import Deps
from agents.orchestrator_agent import orchestrator_agent
from agents.director_agent import director_agent
from agents.github_agent import github_agent
from agents.communication_agent import communication_agent
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
    client: httpx.AsyncClient, step: WorkflowStep, agents: dict
) -> AgentResponse:
    """Execute a single workflow step using the appropriate agent."""
    agent = agents[step.agent_type]

    try:
        result = await agent.run(
            user_prompt=step.step_description,
            deps=getattr(Deps, step.agent_type.upper(), None),
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

    for step in plan_result:
        agent_type = AgentType[step["agent_type"].upper()]

        if current_agent == agent_type:
            current_descriptions.append(step["step_description"])
        else:
            if current_descriptions:
                workflow_step = WorkflowStep(
                    step_description=" AND ".join(current_descriptions),
                    agent_type=current_agent,  # type: ignore
                )
                steps.append(workflow_step)
            current_agent = agent_type
            current_descriptions = [step["step_description"]]

    if current_descriptions:
        workflow_step = WorkflowStep(
            step_description=" AND ".join(current_descriptions),
            agent_type=current_agent,  # type: ignore
        )
        steps.append(workflow_step)

    return steps


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

        # Store user's query
        await supabase_memory.store_message(
            session_id=request.session_id, message_type="human", content=request.query
        )

        # Initialize HTTP client for API calls
        async with httpx.AsyncClient() as client:
            # Initialize agents
            agents = {
                "orchestrator": orchestrator_agent,
                "director": director_agent,
                "github": github_agent,
                "communication": communication_agent,
            }

            # Get initial workflow plan from director
            plan_result = await director_agent.run(
                user_prompt=request.query,
            )
            # Initialize workflow plan
            current_plan = WorkflowPlan(
                plan_id=str(uuid.uuid4()),
                user_query=request.query,
                steps=await create_steps_from_plan(plan_result.data),  # type: ignore
            )

            # Store initial plan
            await supabase_memory.store_plan(
                session_id=request.session_id,
                query=request.query,
                workflow=current_plan.model_dump(),
                results={},
            )

            # Execute workflow steps
            final_result = None
            while current_plan.current_step_index < len(current_plan.steps):
                current_step = current_plan.steps[current_plan.current_step_index]

                # Execute step
                step_result = await execute_workflow_step(client, current_step, agents)
                current_plan.current_step_index += 1

                # Store step result
                # await supabase_memory.store_step_result(
                #     session_id=request.session_id,
                #     step_id=current_step.step_id,
                #     result=step_result,
                # )

            #     # Evaluate result
            #     should_continue = await execute_workflow_step(
            #         client,
            #         WorkflowStep(
            #             step_id=str(uuid.uuid4()),
            #             step_description="Evaluate response",
            #             agent_type="orchestrator",
            #             dependencies={
            #                 "response": step_result,
            #                 "current_plan": current_plan,
            #             },
            #         ),
            #         agents,
            #     )

            #     if not should_continue.result:
            #         # Update plan if needed
            #         current_plan = (
            #             await execute_workflow_step(
            #                 client,
            #                 WorkflowStep(
            #                     step_id=str(uuid.uuid4()),
            #                     step_description="Update workflow plan",
            #                     agent_type="orchestrator",
            #                     dependencies={
            #                         "current_plan": current_plan,
            #                         "response": step_result,
            #                     },
            #                 ),
            #                 agents,
            #             )
            #         ).result

            #         # Store updated plan
            #         await supabase_memory.store_plan(current_plan)

            #     current_plan.current_step_index += 1
            #     final_result = step_result

            # # Store final response
            # await supabase_memory.store_message(
            #     session_id=request.session_id,
            #     message_type="ai",
            #     content=(
            #         str(final_result.result) if final_result else "Workflow completed"
            #     ),
            #     data={"request_id": request.request_id},
            # )

            return AgentResponse(
                success=True,
                result=None,
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
