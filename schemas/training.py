from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional
from enum import Enum

class ModelType(str, Enum):
    LINEAR = "linear"
    LOGISTIC = "logistic"
    SVM = "svm"
    RANDOM_FOREST = "random_forest"
    KMEANS = "kmeans"
    NEURAL_NETWORK = "neural_network"
    CNN = "cnn"
    RNN = "rnn"
    TRANSFORMER = "transformer"

class DataType(str, Enum):
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"

class TrainingRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str = Field(..., description="The name of the model to be trained.")
    model_type: ModelType = Field(..., description="The type of the model.")
    dataset_id: str = Field(..., description="The ID of the dataset to use for training.")
    hyperparameters: Dict[str, Any] = Field({}, description="A dictionary of hyperparameters.")

class TrainingStatus(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    job_id: str = Field(..., description="The unique ID for the training job.")
    status: str = Field(..., description="The current status of the training job.")
    metrics: Dict[str, Any] = Field({}, description="Performance metrics of the trained model.")
    error: Optional[str] = Field(None, description="Error message if the job failed.")
    created_at: str = Field(..., description="The timestamp when the job was created.")
    model_path: Optional[str] = Field(None, description="File path of the trained model.")

class TrainingResponse(BaseModel):
    # This class has no 'model_' prefix fields, so no change is needed.
    job_id: str = Field(..., description="The unique ID for the training job.")
    status: str = Field(..., description="The current status of the training job.")
    message: str = Field(..., description="A message about the training job.")
    iteration_id: Optional[str] = Field(None, description="The ID of the training iteration log.")