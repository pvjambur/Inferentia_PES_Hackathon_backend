from typing import Dict, Any, List, Callable, Union

class State:
    """
    A simple class to manage and hold the state of the LangGraph workflow.
    """
    def __init__(self, initial_state: Dict[str, Any] = {}):
        self._state = initial_state.copy()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a value from the state.
        """
        return self._state.get(key, default)

    def update(self, new_state: Dict[str, Any]):
        """
        Updates the state with new key-value pairs.
        """
        self._state.update(new_state)

    def all(self) -> Dict[str, Any]:
        """
        Returns a copy of the entire state dictionary.
        """
        return self._state.copy()

class Graph:
    """
    Represents the workflow graph, defining nodes and edges.
    """
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.conditional_edges = {}
        self.entry_point = None

    def add_node(self, name: str, agent: Any):
        """
        Adds a node (agent) to the graph.
        """
        if name in self.nodes:
            raise ValueError(f"Node with name '{name}' already exists.")
        self.nodes[name] = agent

    def add_edge(self, start_node: str, end_node: str):
        """
        Adds a direct edge from one node to another.
        """
        if start_node not in self.nodes or end_node not in self.nodes:
            raise ValueError("Start or end node not found in the graph.")
        self.edges[start_node] = end_node

    def add_conditional_edge(self, start_node: str, condition: Callable[[Dict[str, Any]], str]):
        """
        Adds a conditional edge where the next node is determined by a function.
        """
        if start_node not in self.nodes:
            raise ValueError("Start node not found in the graph.")
        self.conditional_edges[start_node] = condition

    def set_entry_point(self, node_name: str):
        """
        Sets the starting node for the graph execution.
        """
        if node_name not in self.nodes:
            raise ValueError(f"Entry point node '{node_name}' not found.")
        self.entry_point = node_name

    def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the graph workflow starting from the entry point.
        """
        if self.entry_point is None:
            raise ValueError("Entry point has not been set.")
        
        state = State(initial_state)
        current_node_name = self.entry_point

        while current_node_name != "end":
            if current_node_name not in self.nodes:
                raise ValueError(f"Node '{current_node_name}' not found during execution.")
            
            agent = self.nodes[current_node_name]
            updated_state_data = agent.execute(state.all())
            state.update(updated_state_data)

            if current_node_name in self.conditional_edges:
                condition_func = self.conditional_edges[current_node_name]
                current_node_name = condition_func(state.all())
            elif current_node_name in self.edges:
                current_node_name = self.edges[current_node_name]
            else:
                current_node_name = "end"
        
        return state.all()