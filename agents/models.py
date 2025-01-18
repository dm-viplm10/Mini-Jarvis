from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime

class AgentType(str, Enum):
    DIRECTOR = "director"
    GITHUB = "github"
    COMMUNICATION = "communication"

class AgentRequest(BaseModel):
    query: str
    user_id: str
    request_id: str
    session_id: str

class AgentAction(BaseModel):
    agent_type: AgentType
    action: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
class AgentResponse(BaseModel):
    success: bool
    
class WorkflowStep(BaseModel):
    agent: AgentType
    action: str
    result: Dict[str, Any]
    timestamp: datetime