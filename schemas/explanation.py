from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional, Union

class ExplanationRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str = Field(..., description="The unique ID of the trained model to explain.")
    data_point: Dict[str, Any] = Field(..., description="The data point for which to generate an explanation.")
    
class FeatureImpact(BaseModel):
    feature_name: str = Field(..., description="The name of the feature.")
    shap_value: float = Field(..., description="The SHAP value representing the feature's impact on the prediction.")
    
class ExplanationResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str = Field(..., description="The ID of the model that was explained.")
    prediction: Any = Field(..., description="The model's prediction for the given data point.")
    base_value: float = Field(..., description="The base value or expected output of the model.")
    feature_impacts: List[FeatureImpact] = Field(..., description="A list of features and their impact on the prediction.")
    visual_explanation: Optional[str] = Field(None, description="A path or reference to a visual explanation (e.g., a SHAP plot).")

class PlotRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str = Field(..., description="The unique ID of the trained model to generate plots for.")
    plot_types: List[str] = Field(..., description="The types of plots to generate (e.g., ['summary', 'waterfall']).")

class PlotResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    plots: Dict[str, str] # Dictionary of plot_type: plot_url
    status: str

class TopFeaturesResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    top_features: List[Dict[str, Union[str, float]]]
    status: str

class ComparisonResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    comparison: Dict[str, Any]
    status: str

class ExplainabilityInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    model_type: str
    supports_shap: bool
    available_explanations: List[str]
    available_plots: List[str]
    explainer_type: str