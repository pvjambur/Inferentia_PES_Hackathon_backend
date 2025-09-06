from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import json
import logging

from database.json_db import JSONDatabase
from core.explanation.shap_explainer import SHAPExplainer
from utils.data_processing import DataProcessor
from schemas.explanation import (
    ExplanationRequest, ExplanationResponse, PlotRequest
)

logger = logging.getLogger(__name__)
router = APIRouter()

async def get_db() -> JSONDatabase:
    from main import app
    return app.state.db

@router.post("/create-explainer/{model_id}")
async def create_explainer(
    model_id: str,
    background_file: UploadFile = File(...),
    db: JSONDatabase = Depends(get_db)
):
    """Create SHAP explainer for a trained model"""
    try:
        # Get model info
        models = await db.get_models()
        if model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
            
        model_info = models[model_id]
        model_path = model_info['file_path']
        
        # Process background data
        processor = DataProcessor()
        
        # Save background file temporarily
        background_path = f"temp_background_{model_id}.csv"
        with open(background_path, "wb") as buffer:
            content = await background_file.read()
            buffer.write(content)
        
        try:
            # Load and process background data
            X_background, _ = await processor.load_and_process(
                background_path, model_info['data_type']
            )
            
            # Create explainer
            explainer = SHAPExplainer()
            result = await explainer.create_explainer(model_path, X_background)
            
            # Store explainer reference (in production, use Redis or similar)
            # For now, we'll create it on-demand
            
            return {
                "model_id": model_id,
                "explainer_status": "created",
                "details": result
            }
            
        finally:
            # Clean up temp file
            import os
            if os.path.exists(background_path):
                os.remove(background_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating explainer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/explain-instance/{model_id}")
async def explain_instance(
    model_id: str,
    instance_data: str = Form(...),
    feature_names: Optional[str] = Form(None),
    db: JSONDatabase = Depends(get_db)
):
    """Explain a single prediction instance"""
    try:
        # Get model info
        models = await db.get_models()
        if model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
            
        model_info = models[model_id]
        model_path = model_info['file_path']
        
        # Parse instance data
        try:
            instance_values = json.loads(instance_data)
            X_instance = np.array(instance_values)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid instance data format")
        
        # Parse feature names
        features = None
        if feature_names:
            try:
                features = json.loads(feature_names)
            except json.JSONDecodeError:
                pass
        
        # Create temporary background data (in production, cache this)
        processor = DataProcessor()
        # For demo, create dummy background data
        background_size = 50
        feature_count = len(X_instance)
        X_background = np.random.randn(background_size, feature_count)
        
        # Create explainer and explain
        explainer = SHAPExplainer()
        await explainer.create_explainer(model_path, X_background)
        explanation = await explainer.explain_instance(X_instance, model_path, features)
        
        return {
            "model_id": model_id,
            "explanation": explanation,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining instance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/explain-dataset/{model_id}")
async def explain_dataset(
    model_id: str,
    dataset_file: UploadFile = File(...),
    feature_names: Optional[str] = Form(None),
    max_samples: int = Form(100),
    db: JSONDatabase = Depends(get_db)
):
    """Explain predictions for a dataset"""
    try:
        # Get model info
        models = await db.get_models()
        if model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
            
        model_info = models[model_id]
        model_path = model_info['file_path']
        
        # Save dataset file temporarily
        dataset_path = f"temp_dataset_{model_id}.csv"
        with open(dataset_path, "wb") as buffer:
            content = await dataset_file.read()
            buffer.write(content)
        
        try:
            # Load and process dataset
            processor = DataProcessor()
            X_data, _ = await processor.load_and_process(
                dataset_path, model_info['data_type']
            )
            
            # Parse feature names
            features = None
            if feature_names:
                try:
                    features = json.loads(feature_names)
                except json.JSONDecodeError:
                    pass
            
            # Create background data
            background_size = min(50, len(X_data) // 2)
            indices = np.random.choice(len(X_data), background_size, replace=False)
            X_background = X_data[indices]
            
            # Create explainer and explain dataset
            explainer = SHAPExplainer()
            await explainer.create_explainer(model_path, X_background)
            explanation = await explainer.explain_dataset(
                X_data, model_path, features, max_samples
            )
            
            return {
                "model_id": model_id,
                "explanation": explanation,
                "status": "success"
            }
            
        finally:
            # Clean up temp file
            import os
            if os.path.exists(dataset_path):
                os.remove(dataset_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-plots/{model_id}")
async def generate_plots(
    model_id: str,
    dataset_file: UploadFile = File(...),
    plot_types: str = Form("summary,waterfall,force"),
    feature_names: Optional[str] = Form(None),
    db: JSONDatabase = Depends(get_db)
):
    """Generate SHAP visualization plots"""
    try:
        # Get model info
        models = await db.get_models()
        if model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
            
        model_info = models[model_id]
        model_path = model_info['file_path']
        
        # Parse plot types
        plot_list = [pt.strip() for pt in plot_types.split(",")]
        
        # Save dataset file temporarily
        dataset_path = f"temp_dataset_{model_id}.csv"
        with open(dataset_path, "wb") as buffer:
            content = await dataset_file.read()
            buffer.write(content)
        
        try:
            # Load and process dataset
            processor = DataProcessor()
            X_data, _ = await processor.load_and_process(
                dataset_path, model_info['data_type']
            )
            
            # Parse feature names
            features = None
            if feature_names:
                try:
                    features = json.loads(feature_names)
                except json.JSONDecodeError:
                    pass
            
            # Create background data
            background_size = min(50, len(X_data) // 2)
            indices = np.random.choice(len(X_data), background_size, replace=False)
            X_background = X_data[indices]
            
            # Create explainer and generate plots
            explainer = SHAPExplainer()
            await explainer.create_explainer(model_path, X_background)
            plots = await explainer.generate_plots(
                X_data, model_path, features, plot_list
            )
            
            return {
                "model_id": model_id,
                "plots": plots,
                "plot_types": plot_list,
                "status": "success"
            }
            
        finally:
            # Clean up temp file
            import os
            if os.path.exists(dataset_path):
                os.remove(dataset_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/top-features/{model_id}")
async def get_top_features(
    model_id: str,
    dataset_file: UploadFile = File(...),
    top_k: int = 10,
    feature_names: Optional[str] = Form(None),
    db: JSONDatabase = Depends(get_db)
):
    """Get top K most important features"""
    try:
        # Get model info
        models = await db.get_models()
        if model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
            
        model_info = models[model_id]
        model_path = model_info['file_path']
        
        # Save dataset file temporarily
        dataset_path = f"temp_dataset_{model_id}.csv"
        with open(dataset_path, "wb") as buffer:
            content = await dataset_file.read()
            buffer.write(content)
        
        try:
            # Load and process dataset
            processor = DataProcessor()
            X_data, _ = await processor.load_and_process(
                dataset_path, model_info['data_type']
            )
            
            # Parse feature names
            features = None
            if feature_names:
                try:
                    features = json.loads(feature_names)
                except json.JSONDecodeError:
                    pass
            
            # Create background data
            background_size = min(50, len(X_data) // 2)
            indices = np.random.choice(len(X_data), background_size, replace=False)
            X_background = X_data[indices]
            
            # Create explainer and get top features
            explainer = SHAPExplainer()
            await explainer.create_explainer(model_path, X_background)
            top_features = await explainer.get_top_features(
                X_data, model_path, features, top_k
            )
            
            return {
                "model_id": model_id,
                "top_features": top_features,
                "top_k": top_k,
                "status": "success"
            }
            
        finally:
            # Clean up temp file
            import os
            if os.path.exists(dataset_path):
                os.remove(dataset_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting top features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare-predictions/{model_id}")
async def compare_predictions(
    model_id: str,
    original_instance: str = Form(...),
    modified_instance: str = Form(...),
    feature_names: Optional[str] = Form(None),
    db: JSONDatabase = Depends(get_db)
):
    """Compare SHAP explanations between original and modified instances"""
    try:
        # Get model info
        models = await db.get_models()
        if model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
            
        model_info = models[model_id]
        model_path = model_info['file_path']
        
        # Parse instance data
        try:
            original_values = json.loads(original_instance)
            modified_values = json.loads(modified_instance)
            X_original = np.array(original_values)
            X_modified = np.array(modified_values)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid instance data format")
        
        # Parse feature names
        features = None
        if feature_names:
            try:
                features = json.loads(feature_names)
            except json.JSONDecodeError:
                pass
        
        # Create background data
        background_size = 50
        feature_count = len(X_original)
        X_background = np.random.randn(background_size, feature_count)
        
        # Create explainer and compare
        explainer = SHAPExplainer()
        await explainer.create_explainer(model_path, X_background)
        comparison = await explainer.explain_prediction_change(
            X_original, X_modified, model_path, features
        )
        
        return {
            "model_id": model_id,
            "comparison": comparison,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-explainability/{model_id}")
async def get_model_explainability_info(
    model_id: str,
    db: JSONDatabase = Depends(get_db)
):
    """Get explainability information for a model"""
    try:
        # Get model info
        models = await db.get_models()
        if model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
            
        model_info = models[model_id]
        
        # Determine explainability features based on model type
        explainability_info = {
            "model_id": model_id,
            "model_type": model_info['model_type'],
            "supports_shap": True,  # All our ML models support SHAP
            "available_explanations": [
                "instance_explanation",
                "dataset_explanation", 
                "feature_importance",
                "prediction_comparison"
            ],
            "available_plots": [
                "summary_plot",
                "waterfall_plot", 
                "force_plot"
            ],
            "explainer_types": {
                "linear": "LinearExplainer",
                "logistic": "LinearExplainer", 
                "svm_classifier": "KernelExplainer",
                "svm_regressor": "KernelExplainer",
                "random_forest_classifier": "TreeExplainer",
                "random_forest_regressor": "TreeExplainer",
                "kmeans": "KernelExplainer"
            }.get(model_info['model_type'], "KernelExplainer")
        }
        
        return explainability_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting explainability info: {e}")
        raise HTTPException(status_code=500, detail=str(e))