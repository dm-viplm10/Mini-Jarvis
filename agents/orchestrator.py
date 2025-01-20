from agents import github_agent, director_agent
from agents.models import AgentType, WorkflowStep
from db.supabase_client import SupabaseMemory
from datetime import datetime
from typing import List, Dict, Any

class Orchestrator:
    def __init__(self):
        self.director = director_agent
        self.github_agent = github_agent
        # self.communication_agent = CommunicationAgent()
        self.memory = SupabaseMemory()
        
    async def process_query(self, query: str) -> Dict[str, Any]:
        # Find similar past executions
        similar_plans = await self.memory.find_similar_plans(query)
        
        # Get workflow plan from director
        actions = await self.director.plan_workflow(query)
        
        # Execute workflow
        workflow_steps: List[WorkflowStep] = []
        results: Dict[str, Any] = {}
        
        for action in actions:
            step_result = None
            
            if action.agent_type == AgentType.GITHUB:
                step_result = await self.github_agent.execute_action(action)
            elif action.agent_type == AgentType.COMMUNICATION:
                step_result = await self.communication_agent.execute_action(action)
                
            if step_result:
                workflow_steps.append(
                    WorkflowStep(
                        agent=action.agent_type,
                        action=action.action,
                        result=step_result.data,
                        timestamp=datetime.utcnow()
                    )
                )
                results[f"{action.agent_type}_{action.action}"] = step_result.data
        
        # Store execution plan
        await self.memory.store_plan(
            query=query,
            workflow=[step.dict() for step in workflow_steps],
            results=results
        )
        
        return {
            "workflow": workflow_steps,
            "results": results,
            "similar_plans": similar_plans
        }