from openai import OpenAI
from config import openrouter_api_key, openrouter_base_url
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from models.agent_workflows import (
    WorkflowPlan,
    AgentResponse,
)
from models.agent_dependencies import OrchestratorDeps


system_prompt = """You are an Orchestrator agent responsible for managing and evaluating multi-agent workflows.
Your responsibilities include:
1. Evaluating agent responses against the workflow plan
2. Deciding if the workflow needs modification based on results
3. Updating the workflow plan when necessary
4. Ensuring the workflow stays on track towards the user's goal

You work with:
1. Director Agent: Creates workflow plans
2. Github Agent: Handles github repository operations
3. Communication Agent: Manages emails and calendar

Make decisions based on the overall context and goal of the workflow."""

orchestrator_agent = Agent(
    model=OpenAIModel(
        "google/gemini-2.0-flash-exp:free",
        api_key=openrouter_api_key,
        base_url=openrouter_base_url,
    ),
    system_prompt=system_prompt,
    deps_type=OrchestratorDeps,
    retries=2,
)


@orchestrator_agent.tool
async def evaluate_response(
    ctx: RunContext[OrchestratorDeps],
    response: AgentResponse,
    current_plan: WorkflowPlan,
) -> bool:
    """Evaluate an agent's response and decide if the workflow should continue as planned."""
    llm = OpenAI(
        api_key=openrouter_api_key,
        base_url=openrouter_base_url,
    )
    evaluation = llm.chat.completions.create(
        model="google/gemini-2.0-flash-exp:free",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
            Evaluate this agent response against the current workflow plan:
            
            Current Step: {current_plan.steps[current_plan.current_step_index].model_dump()}
            Agent Response: {response.model_dump()}
            
            Determine if:
            1. The response successfully achieved the step's expected output
            2. The workflow can continue as planned
            3. Any modifications are needed
            
            Return true if the workflow should continue as is, false if it needs modification.
            """,
            },
        ],
    )

    return True


@orchestrator_agent.tool
async def update_plan(
    ctx: RunContext[OrchestratorDeps],
    current_plan: WorkflowPlan,
    response: AgentResponse,
) -> WorkflowPlan:
    """Update the workflow plan based on an agent's response."""
    updated_plan = await ctx.model.chat(
        messages=[
            {"role": "system", "content": ctx.agent.system_prompt},
            {
                "role": "user",
                "content": f"""
            The current workflow plan needs modification based on this agent response:
            
            Current Plan: {current_plan.model_dump()}
            Agent Response: {response.model_dump()}
            
            Please provide an updated workflow plan that:
            1. Addresses any issues or new information from the response
            2. Maintains progress towards the original goal
            3. Specifies any new or modified steps needed
            
            Return the complete updated plan in a format that can be parsed into a WorkflowPlan object.
            """,
            },
        ]
    )

    # Parse updated plan
    # This is a placeholder - you'll need to implement proper parsing based on LLM output format
    new_plan = current_plan  # Modify based on LLM response

    # Store updated plan
    await ctx.deps.memory.update_plan(new_plan)

    return new_plan
