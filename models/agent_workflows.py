import uuid
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum


class AgentType(str, Enum):
    ORCHESTRATOR = "orchestrator"
    DIRECTOR = "director"
    GITHUB = "github"
    WEB_RESEARCH = "web_research"


class AgentRequest(BaseModel):
    query: str
    user_id: str
    request_id: str
    session_id: str


class AgentActions(BaseModel):
    agent_type: AgentType
    actions: List[str]
    parameters: Dict[str, Any] = Field(default_factory=dict)


class WorkflowStepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowStep(BaseModel):
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_description: str
    agent_type: AgentType
    inputs: Dict[str, Any] = Field(default_factory=dict)
    status: WorkflowStepStatus = WorkflowStepStatus.PENDING
    parent_step_id: Optional[str] = None
    next_step_ids: List[str] = []


class WorkflowPlan(BaseModel):
    plan_id: str
    user_query: str
    steps: List[WorkflowStep]
    current_step_index: int = 0
    status: WorkflowStepStatus = WorkflowStepStatus.IN_PROGRESS


class AgentResponse(BaseModel):
    agent_type: AgentType
    success: bool
    result: Any
    error: Optional[str] = None
