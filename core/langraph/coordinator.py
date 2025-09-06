from typing import Dict, Any, List, Callable, Union
from utils.logging import get_logger
from core.langraph.agents import HumanAgent, MLTrainerAgent, DataProcessingAgent

# Corrected imports from the official langgraph library
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages

logger = get_logger(__name__)

# Define the state schema using TypedDict and Annotated
class WorkflowState(TypedDict):
    """State for our graph, a dict of attributes."""
    # This key will hold the state of our workflow, with add_messages
    # ensuring new updates append to the list.
    data: Annotated[list, add_messages]


class LangGraphCoordinator:
    """
    Coordinates and executes a LangGraph workflow.
    This class orchestrates different agents to perform a multi-step task.
    """
    def __init__(self):
        self.agents = {
            "human_agent": HumanAgent(),
            "ml_trainer_agent": MLTrainerAgent(),
            "data_processing_agent": DataProcessingAgent()
        }
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Defines the structure of the LangGraph workflow using the official library.
        """
        # Use the official StateGraph class
        graph_builder = StateGraph(WorkflowState)

        # Define the nodes (agents) and their functions
        graph_builder.add_node("process_data", self.agents["data_processing_agent"].execute)
        graph_builder.add_node("train_model", self.agents["ml_trainer_agent"].execute)
        graph_builder.add_node("get_user_feedback", self.agents["human_agent"].execute)
        
        # Define the edges (flow of control)
        graph_builder.add_edge(START, "get_user_feedback")
        graph_builder.add_edge("get_user_feedback", "process_data")
        graph_builder.add_edge("process_data", "train_model")
        
        # A conditional edge for re-training or finishing
        def should_continue(state: Dict[str, Any]):
            if state.get("needs_retraining"):
                return "get_user_feedback"
            else:
                return "end"

        graph_builder.add_conditional_edges("train_model", should_continue, {
            "get_user_feedback": "get_user_feedback",
            "end": END
        })
        
        # Compile the graph
        return graph_builder.compile()

    def run_agent_task(self, task_id: str, initial_state: Dict[str, Any]):
        """
        Executes a LangGraph task from a given initial state.
        """
        logger.info(f"Starting LangGraph task {task_id} with initial state: {initial_state}")
        
        try:
            # The `invoke` method of the CompiledGraph executes the defined flow.
            # It returns the final state directly.
            final_state = self.graph.invoke(initial_state)
            logger.info(f"LangGraph task {task_id} completed successfully.")
            return final_state
        except Exception as e:
            logger.error(f"LangGraph task {task_id} failed: {e}")
            raise