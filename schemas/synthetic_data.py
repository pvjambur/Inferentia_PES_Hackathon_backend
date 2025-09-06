from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from enum import Enum

class DataType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"

class SyntheticDataRequest(BaseModel):
    # No 'model_' prefix fields here, so no change is needed.
    data_type: DataType = Field(..., description="The type of data to generate (e.g., 'text', 'image', 'audio').")
    num_samples: int = Field(..., gt=0, description="The number of synthetic samples to generate.")
    output_directory: Optional[str] = Field(None, description="The directory to save the generated data. If not provided, a default location will be used.")
    generation_params: Dict[str, Any] = Field({}, description="Additional parameters for the generation process.")

class SyntheticDataResponse(BaseModel):
    # No 'model_' prefix fields here, so no change is needed.
    status: str = Field(..., description="The status of the generation task (e.g., 'success', 'failed').")
    generated_file_paths: List[str] = Field(..., description="A list of file paths where the generated data is stored.")
    details: Optional[str] = Field(None, description="Additional details about the generation process.")