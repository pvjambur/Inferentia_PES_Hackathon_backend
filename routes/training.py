from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks, Form
from fastapi.responses import FileResponse
import os
import json
import uuid
from typing import List, Optional, Dict, Any
import logging

from database.json_db import JSONDatabase
from core.training.ml_trainer import MLTrainer
from core.training.dl_trainer import DLTrainer
from utils.data_processing import DataProcessor
from schemas.training import (
    TrainingStatus, TrainingResponse, ModelType, DataType
)

logger = logging.getLogger(__name__)
router = APIRouter()

async def get_db() -> JSONDatabase:
    from main import app
    return app.state.db

@router.post("/start", response_model=TrainingResponse)
async def start_training(
    background_tasks: BackgroundTasks,
    agent_id: str = Form(...),
    model_type: ModelType = Form(...),
    data_type: DataType = Form(...),
    dataset_file: UploadFile = File(...),
    hyperparameters: str = Form("{}"),
    db: JSONDatabase = Depends(get_db)
):
    """Start model training"""
    try:
        # Parse hyperparameters
        try:
            hyperparams = json.loads(hyperparameters)
        except json.JSONDecodeError:
            hyperparams = {}
            
        # Generate training job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded dataset
        dataset_path = await save_dataset(dataset_file, data_type, job_id)
        
        # Log training iteration
        iteration_data = {
            'agent_id': agent_id,
            'job_id': job_id,
            'model_type': model_type.value,
            'data_type': data_type.value,
            'dataset_path': dataset_path,
            'hyperparameters': hyperparams,
            'status': 'started'
        }
        
        iteration_id = await db.log_iteration(iteration_data)
        
        # Start training in background
        if model_type in [ModelType.LINEAR, ModelType.LOGISTIC, ModelType.SVM, ModelType.RANDOM_FOREST, ModelType.KMEANS]:
            background_tasks.add_task(
                train_ml_model, job_id, agent_id, model_type, data_type, 
                dataset_path, hyperparams, db
            )
        else:
            background_tasks.add_task(
                train_dl_model, job_id, agent_id, model_type, data_type, 
                dataset_path, hyperparams, db
            )
        
        return TrainingResponse(
            job_id=job_id,
            status="started",
            message="Training started successfully",
            iteration_id=iteration_id
        )
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def save_dataset(file: UploadFile, data_type: DataType, job_id: str) -> str:
    """Save uploaded dataset file"""
    try:
        # Create directory structure
        data_dir = f"data/datasets/{data_type.value.lower()}"
        os.makedirs(data_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(data_dir, f"{job_id}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        return file_path
        
    except Exception as e:
        logger.error(f"Error saving dataset: {e}")
        raise

async def train_ml_model(
    job_id: str, agent_id: str, model_type: ModelType, data_type: DataType,
    dataset_path: str, hyperparams: dict, db: JSONDatabase
):
    """Train ML model in background"""
    try:
        # Update status
        await db.log_iteration({
            'job_id': job_id,
            'agent_id': agent_id,
            'status': 'training'
        })
        
        # Initialize trainer
        trainer = MLTrainer()
        
        # Process data
        processor = DataProcessor()
        X, y = await processor.load_and_process(dataset_path, data_type)
        
        # Train model
        model_info = await trainer.train(
            X, y, model_type.value, hyperparams, job_id
        )
        
        # Save model metadata
        await db.create_model(job_id, {
            'agent_id': agent_id,
            'model_type': model_type.value,
            'data_type': data_type.value,
            'file_path': model_info['file_path'],
            'metrics': model_info['metrics'],
            'hyperparameters': hyperparams,
            'status': 'completed'
        })
        
        # Update iteration
        await db.log_iteration({
            'job_id': job_id,
            'agent_id': agent_id,
            'status': 'completed',
            'metrics': model_info['metrics'],
            'model_path': model_info['file_path']
        })
        
        logger.info(f"ML training completed for job {job_id}")
        
    except Exception as e:
        logger.error(f"ML training failed for job {job_id}: {e}")
        await db.log_iteration({
            'job_id': job_id,
            'agent_id': agent_id,
            'status': 'failed',
            'error': str(e)
        })

async def train_dl_model(
    job_id: str, agent_id: str, model_type: ModelType, data_type: DataType,
    dataset_path: str, hyperparams: dict, db: JSONDatabase
):
    """Train DL model in background"""
    try:
        # Update status
        await db.log_iteration({
            'job_id': job_id,
            'agent_id': agent_id,
            'status': 'training'
        })
        
        # Initialize trainer
        trainer = DLTrainer()
        
        # Process data
        processor = DataProcessor()
        data_loader = await processor.create_data_loader(dataset_path, data_type)
        
        # Train model
        model_info = await trainer.train(
            data_loader, model_type.value, hyperparams, job_id
        )
        
        # Save model metadata
        await db.create_model(job_id, {
            'agent_id': agent_id,
            'model_type': model_type.value,
            'data_type': data_type.value,
            'file_path': model_info['file_path'],
            'metrics': model_info['metrics'],
            'hyperparameters': hyperparams,
            'status': 'completed'
        })
        
        # Update iteration
        await db.log_iteration({
            'job_id': job_id,
            'agent_id': agent_id,
            'status': 'completed',
            'metrics': model_info['metrics'],
            'model_path': model_info['file_path']
        })
        
        logger.info(f"DL training completed for job {job_id}")
        
    except Exception as e:
        logger.error(f"DL training failed for job {job_id}: {e}")
        await db.log_iteration({
            'job_id': job_id,
            'agent_id': agent_id,
            'status': 'failed',
            'error': str(e)
        })

@router.get("/status/{job_id}", response_model=TrainingStatus)
async def get_training_status(job_id: str, db: JSONDatabase = Depends(get_db)):
    """Get training job status"""
    try:
        iterations = await db.get_iterations()
        job_iterations = [i for i in iterations if i.get('job_id') == job_id]
        
        if not job_iterations:
            raise HTTPException(status_code=404, detail="Job not found")
            
        latest = max(job_iterations, key=lambda x: x.get('created_at', '')) # Use get for robustness
        return {
            'job_id': job_id,
            'status': latest.get('status', 'unknown'),
            'metrics': latest.get('metrics', {}),
            'error': latest.get('error'),
            'created_at': latest['created_at'],
            'model_path': latest.get('model_path')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{agent_id}")
async def get_training_history(agent_id: str, db: JSONDatabase = Depends(get_db)):
    """Get training history for agent"""
    try:
        iterations = await db.get_iterations(agent_id)
        return {
            'agent_id': agent_id,
            'iterations': iterations
        }
    except Exception as e:
        logger.error(f"Error getting training history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{agent_id}")
async def get_agent_models(agent_id: str, db: JSONDatabase = Depends(get_db)):
    """Get all models for an agent"""
    try:
        models = await db.get_models()
        agent_models = {k: v for k, v in models.items() if v.get('agent_id') == agent_id}
        return {
            'agent_id': agent_id,
            'models': agent_models
        }
    except Exception as e:
        logger.error(f"Error getting agent models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/model/{model_id}")
async def delete_model(model_id: str, db: JSONDatabase = Depends(get_db)):
    """Delete a trained model"""
    try:
        models = await db.get_models()
        if model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
            
        model_info = models[model_id]
        
        # Delete model file
        if os.path.exists(model_info['file_path']):
            os.remove(model_info['file_path'])
            
        # Remove from database
        del models[model_id]
        file_path = os.path.join(db.base_path, db.files['models'])
        await db._write_file(file_path, models)
        
        return {"message": f"Model {model_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrain/{model_id}")
async def retrain_model(
    model_id: str,
    background_tasks: BackgroundTasks,
    dataset_file: Optional[UploadFile] = File(None),
    hyperparameters: str = Form("{}"),
    db: JSONDatabase = Depends(get_db)
):
    """Retrain existing model with new data or hyperparameters"""
    try:
        models = await db.get_models()
        if model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
            
        model_info = models[model_id]
        
        # Parse hyperparameters
        try:
            hyperparams = json.loads(hyperparameters)
        except json.JSONDecodeError:
            hyperparams = model_info['hyperparameters']
            
        # Generate new job ID for retraining
        job_id = str(uuid.uuid4())
        
        # Use new dataset if provided, otherwise use original
        if dataset_file:
            dataset_path = await save_dataset(dataset_file, model_info['data_type'], job_id)
        else:
            # Find original dataset path from iterations
            iterations = await db.get_iterations(model_info['agent_id'])
            original_iter = next((i for i in iterations if i.get('job_id') == model_id), None)
            dataset_path = original_iter['dataset_path'] if original_iter else None
            
        if not dataset_path:
            raise HTTPException(status_code=400, detail="No dataset available for retraining")
            
        # Log retraining iteration
        iteration_data = {
            'agent_id': model_info['agent_id'],
            'job_id': job_id,
            'model_type': model_info['model_type'],
            'data_type': model_info['data_type'],
            'dataset_path': dataset_path,
            'hyperparameters': hyperparams,
            'status': 'started',
            'retrain_of': model_id
        }
        
        iteration_id = await db.log_iteration(iteration_data)
        
        # Start retraining in background
        if model_info['model_type'] in [ModelType.LINEAR.value, ModelType.LOGISTIC.value, ModelType.SVM.value, ModelType.RANDOM_FOREST.value, ModelType.KMEANS.value]:
            background_tasks.add_task(
                train_ml_model, job_id, model_info['agent_id'], 
                model_info['model_type'], model_info['data_type'],
                dataset_path, hyperparams, db
            )
        else:
            background_tasks.add_task(
                train_dl_model, job_id, model_info['agent_id'],
                model_info['model_type'], model_info['data_type'],
                dataset_path, hyperparams, db
            )
        
        return TrainingResponse(
            job_id=job_id,
            status="started",
            message="Retraining started successfully",
            iteration_id=iteration_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))
