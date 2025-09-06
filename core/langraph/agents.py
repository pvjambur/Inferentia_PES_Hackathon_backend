from typing import Dict, Any, List
from utils.logging import get_logger

logger = get_logger(__name__)

class BaseAgent:
    """
    Base class for a LangGraph agent node.
    """
    def __init__(self, name: str):
        self.name = name

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the agent's logic based on the current state and returns an updated state.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the 'execute' method.")

class HumanAgent(BaseAgent):
    """
    Simulates a human or a user interaction step in the graph.
    """
    def __init__(self):
        super().__init__(name="HumanAgent")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Executing HumanAgent. Current state: {state}")
        
        state['user_confirmation'] = True # Assume user confirms
        
        logger.info("HumanAgent executed. User confirmed.")
        return state

class MLTrainerAgent(BaseAgent):
    """
    An agent responsible for training an ML model.
    """
    def __init__(self):
        super().__init__(name="MLTrainerAgent")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Executing MLTrainerAgent. Current state: {state}")
        
        state['model_trained'] = True
        state['model_path'] = "/path/to/trained_model.pkl"
        state['training_metrics'] = {"accuracy": 0.95, "loss": 0.05}
        
        state['needs_retraining'] = False
        
        logger.info("MLTrainerAgent executed. Model trained successfully.")
        return state

class DataProcessingAgent(BaseAgent):
    """
    An agent responsible for data preprocessing tasks.
    """
    def __init__(self):
        super().__init__(name="DataProcessingAgent")
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Executing DataProcessingAgent. Current state: {state}")
        
        state['data_processed'] = True
        state['processed_data_path'] = "/path/to/processed_data.csv"
        
        logger.info("DataProcessingAgent executed. Data processed successfully.")
        return state