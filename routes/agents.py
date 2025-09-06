# Create a new file for this. For example, in your routes directory.
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import json

# Assuming these imports are correct based on your project structure
from database.json_db import JSONDatabase
from database.models import AgentConfig
from core.langraph.coordinator import LangGraphCoordinator

router = APIRouter()

# Global coordinator instance
coordinator = LangGraphCoordinator()

# Dependency to get the database instance
# We'll initialize the database here, in your main.py
def get_db_conn() -> JSONDatabase:
    # This will be overridden in main.py, but it makes the route file self-contained
    # for local testing and clarity.
    db = JSONDatabase("data/agents.json")
    return db

@router.post("/agents/", response_model=AgentConfig, status_code=201)
async def create_agent_config(agent_data: AgentConfig, db: JSONDatabase = Depends(get_db_conn)):
    """
    Creates a new agent configuration.
    """
    try:
        # Use a consistent key for the record, e.g., the agent's ID
        db.create_record(agent_data.agent_id, agent_data.model_dump())
        return agent_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {e}")

@router.get("/agents/", response_model=List[AgentConfig])
async def get_all_agents(db: JSONDatabase = Depends(get_db_conn)):
    """
    Retrieves all agent configurations.
    """
    try:
        agents_data = db.read_all()
        return [AgentConfig(**agent) for agent in agents_data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agents: {e}")

@router.get("/agents/{agent_id}", response_model=AgentConfig)
async def get_agent_by_id(agent_id: str, db: JSONDatabase = Depends(get_db_conn)):
    """
    Retrieves a single agent configuration by its ID.
    """
    try:
        agent_data = db.read_record(agent_id)
        if not agent_data:
            raise HTTPException(status_code=404, detail="Agent not found")
        return AgentConfig(**agent_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agent: {e}")

@router.post("/agents/{agent_id}/run", response_model=Dict[str, Any])
async def run_agent_task(agent_id: str, task_data: Dict[str, Any], db: JSONDatabase = Depends(get_db_conn)):
    """
    Executes a task using a specific domain-specific agent.
    """
    try:
        agent_config = db.read_record(agent_id)
        if not agent_config or not agent_config.get("is_active"):
            raise HTTPException(status_code=404, detail="Agent not found or is inactive.")
        
        # The coordinator will handle the actual task execution
        result = coordinator.run_agent_task(agent_id, task_data)
        return {"status": "success", "result": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run agent task: {e}")