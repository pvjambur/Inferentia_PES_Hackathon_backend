from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from datetime import datetime

class AgentConfig(BaseModel):
    # Add this line to disable the protected namespace check for this model
    model_config = ConfigDict(protected_namespaces=())

    name: str = Field(..., description="The name of the agent.")
    description: str = Field(..., description="A brief description of the agent's purpose.")
    endpoint: str = Field(..., description="The API endpoint associated with the agent.")
    model_id: str = Field(..., description="The ID of the model used by the agent.")
    version: str = Field(..., description="The version of the agent configuration.")
    is_active: bool = Field(True, description="Indicates if the agent is currently active.")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the agent configuration was created.")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the last update to the agent configuration.")
    params: Dict[str, Any] = Field({}, description="Additional parameters for the agent's behavior.")

class ModelMetadata(BaseModel):
    # Add this line as this class has conflicting fields like 'model_type'
    model_config = ConfigDict(protected_namespaces=())

    id: str = Field(..., description="Unique identifier for the trained model.")
    name: str = Field(..., description="The name of the model.")
    model_type: str = Field(..., description="The type of the model (e.g., 'ML', 'DL', 'GAN').")
    dataset_id: str = Field(..., description="The ID of the dataset used for training.")
    training_status: str = Field("untrained", description="Current training status (e.g., 'untrained', 'training', 'trained', 'failed').")
    performance_metrics: Dict[str, Any] = Field({}, description="Performance metrics of the trained model.")
    path: str = Field(..., description="The file path where the model is saved.")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the model metadata was created.")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the last update to the model metadata.")

class DatasetMetadata(BaseModel):
    id: str = Field(..., description="Unique identifier for the dataset.")
    name: str = Field(..., description="The name of the dataset.")
    description: str = Field(..., description="A brief description of the dataset.")
    file_path: str = Field(..., description="The file path where the dataset is stored.")
    file_type: str = Field(..., description="The type of the dataset file (e.g., 'csv', 'json', 'h5').")
    num_samples: Optional[int] = Field(None, description="The number of samples in the dataset.")
    data_schema: Dict[str, Any] = Field({}, description="A schema describing the columns/features of the dataset.")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the dataset metadata was created.")

class TrainingIteration(BaseModel):
    # Add this line as this class has a conflicting field 'model_id'
    model_config = ConfigDict(protected_namespaces=())

    id: str = Field(..., description="Unique identifier for the training iteration.")
    model_id: str = Field(..., description="The ID of the model being trained.")
    start_time: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the training started.")
    end_time: Optional[datetime] = Field(None, description="Timestamp when the training finished.")
    duration_seconds: Optional[int] = Field(None, description="Total duration of the training in seconds.")
    status: str = Field("in_progress", description="Status of the iteration (e.g., 'in_progress', 'completed', 'failed').")
    hyperparameters: Dict[str, Any] = Field({}, description="Hyperparameters used for this training run.")
    metrics: Dict[str, Any] = Field({}, description="Metrics logged during training (e.g., loss, accuracy).")
    logs: Optional[str] = Field(None, description="A path or reference to the training logs.")