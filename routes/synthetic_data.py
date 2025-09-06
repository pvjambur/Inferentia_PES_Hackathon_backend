from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from typing import List, Optional, Dict, Any
import json
import logging

from database.json_db import JSONDatabase
from core.synthetic_data.text_generator import TextSyntheticGenerator
from core.synthetic_data.image_gan import ImageGANGenerator
from core.synthetic_data.audio_gan import AudioGANGenerator
from schemas.synthetic_data import (
    SyntheticDataRequest, SyntheticDataResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()

async def get_db() -> JSONDatabase:
    from main import app
    return app.state.db

@router.post("/generate-text")
async def generate_text_data(
    background_tasks: BackgroundTasks,
    agent_id: str = Form(...),
    dataset_id: str = Form(...),
    num_samples: int = Form(100),
    generation_method: str = Form("groq"),
    feedback_data: Optional[str] = Form(None),
    db: JSONDatabase = Depends(get_db)
):
    """Generate synthetic text data"""
    try:
        # Parse feedback data if provided
        feedback = None
        if feedback_data:
            try:
                feedback = json.loads(feedback_data)
            except json.JSONDecodeError:
                logger.warning("Invalid feedback data format")
        
        # Get dataset info
        datasets = await db.get_datasets()
        if dataset_id not in datasets:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset_info = datasets[dataset_id]
        
        # Start generation in background
        background_tasks.add_task(
            generate_text_synthetic_data,
            agent_id, dataset_id, num_samples, generation_method, 
            feedback, dataset_info, db
        )
        
        return SyntheticDataResponse(
            status="started",
            message=f"Text data generation started for {num_samples} samples",
            generation_type="text",
            dataset_id=dataset_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting text generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-images")
async def generate_image_data(
    background_tasks: BackgroundTasks,
    agent_id: str = Form(...),
    dataset_id: str = Form(...),
    num_samples: int = Form(50),
    gan_epochs: int = Form(1000),
    feedback_data: Optional[str] = Form(None),
    db: JSONDatabase = Depends(get_db)
):
    """Generate synthetic image data using GAN"""
    try:
        # Parse feedback data if provided
        feedback = None
        if feedback_data:
            try:
                feedback = json.loads(feedback_data)
            except json.JSONDecodeError:
                logger.warning("Invalid feedback data format")
        
        # Get dataset info
        datasets = await db.get_datasets()
        if dataset_id not in datasets:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset_info = datasets[dataset_id]
        
        # Start generation in background
        background_tasks.add_task(
            generate_image_synthetic_data,
            agent_id, dataset_id, num_samples, gan_epochs,
            feedback, dataset_info, db
        )
        
        return SyntheticDataResponse(
            status="started",
            message=f"Image GAN training and generation started for {num_samples} samples",
            generation_type="image",
            dataset_id=dataset_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting image generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-audio")
async def generate_audio_data(
    background_tasks: BackgroundTasks,
    agent_id: str = Form(...),
    dataset_id: str = Form(...),
    num_samples: int = Form(30),
    gan_epochs: int = Form(1500),
    feedback_data: Optional[str] = Form(None),
    db: JSONDatabase = Depends(get_db)
):
    """Generate synthetic audio/spectrogram data using GAN"""
    try:
        # Parse feedback data if provided
        feedback = None
        if feedback_data:
            try:
                feedback = json.loads(feedback_data)
            except json.JSONDecodeError:
                logger.warning("Invalid feedback data format")
        
        # Get dataset info
        datasets = await db.get_datasets()
        if dataset_id not in datasets:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset_info = datasets[dataset_id]
        
        # Start generation in background
        background_tasks.add_task(
            generate_audio_synthetic_data,
            agent_id, dataset_id, num_samples, gan_epochs,
            feedback, dataset_info, db
        )
        
        return SyntheticDataResponse(
            status="started",
            message=f"Audio GAN training and generation started for {num_samples} samples",
            generation_type="audio",
            dataset_id=dataset_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting audio generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_text_synthetic_data(
    agent_id: str, dataset_id: str, num_samples: int, 
    generation_method: str, feedback: Optional[Dict], 
    dataset_info: Dict, db: JSONDatabase
):
    """Generate text synthetic data in background"""
    try:
        logger.info(f"Starting text data generation for agent {agent_id}")
        
        # Initialize text generator
        generator = TextSyntheticGenerator()
        
        # Load original dataset
        original_data = await generator.load_original_data(dataset_info['file_path'])
        
        # Generate synthetic data
        synthetic_data = await generator.generate_synthetic_data(
            original_data, num_samples, generation_method, feedback
        )
        
        # Save synthetic data
        synthetic_file_path = await generator.save_synthetic_data(
            synthetic_data, agent_id, dataset_id
        )
        
        # Update dataset with synthetic data
        await db.update_dataset(dataset_id, {
            'synthetic_data_path': synthetic_file_path,
            'synthetic_samples_count': len(synthetic_data),
            'last_synthetic_generation': 'now'
        })
        
        # Log generation iteration
        await db.log_iteration({
            'agent_id': agent_id,
            'dataset_id': dataset_id,
            'type': 'synthetic_generation',
            'data_type': 'text',
            'samples_generated': len(synthetic_data),
            'generation_method': generation_method,
            'status': 'completed'
        })
        
        logger.info(f"Text generation completed: {len(synthetic_data)} samples")
        
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        await db.log_iteration({
            'agent_id': agent_id,
            'dataset_id': dataset_id,
            'type': 'synthetic_generation',
            'data_type': 'text',
            'status': 'failed',
            'error': str(e)
        })

async def generate_image_synthetic_data(
    agent_id: str, dataset_id: str, num_samples: int,
    gan_epochs: int, feedback: Optional[Dict],
    dataset_info: Dict, db: JSONDatabase
):
    """Generate image synthetic data using GAN in background"""
    try:
        logger.info(f"Starting image GAN generation for agent {agent_id}")
        
        # Initialize image GAN
        generator = ImageGANGenerator()
        
        # Load original dataset
        original_images = await generator.load_image_dataset(dataset_info['file_path'])
        
        # Train GAN
        gan_model = await generator.train_gan(
            original_images, gan_epochs, feedback
        )
        
        # Generate synthetic images
        synthetic_images = await generator.generate_images(
            gan_model, num_samples
        )
        
        # Save synthetic images
        synthetic_dir = await generator.save_synthetic_images(
            synthetic_images, agent_id, dataset_id
        )
        
        # Update dataset
        await db.update_dataset(dataset_id, {
            'synthetic_data_path': synthetic_dir,
            'synthetic_samples_count': len(synthetic_images),
            'gan_epochs': gan_epochs,
            'last_synthetic_generation': 'now'
        })
        
        # Log generation iteration
        await db.log_iteration({
            'agent_id': agent_id,
            'dataset_id': dataset_id,
            'type': 'synthetic_generation',
            'data_type': 'image',
            'samples_generated': len(synthetic_images),
            'gan_epochs': gan_epochs,
            'status': 'completed'
        })
        
        logger.info(f"Image GAN generation completed: {len(synthetic_images)} samples")
        
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        await db.log_iteration({
            'agent_id': agent_id,
            'dataset_id': dataset_id,
            'type': 'synthetic_generation',
            'data_type': 'image',
            'status': 'failed',
            'error': str(e)
        })

async def generate_audio_synthetic_data(
    agent_id: str, dataset_id: str, num_samples: int,
    gan_epochs: int, feedback: Optional[Dict],
    dataset_info: Dict, db: JSONDatabase
):
    """Generate audio synthetic data using GAN in background"""
    try:
        logger.info(f"Starting audio GAN generation for agent {agent_id}")
        
        # Initialize audio GAN
        generator = AudioGANGenerator()
        
        # Load original dataset
        original_audio = await generator.load_audio_dataset(dataset_info['file_path'])
        
        # Train GAN on spectrograms
        gan_model = await generator.train_spectrogram_gan(
            original_audio, gan_epochs, feedback
        )
        
        # Generate synthetic spectrograms
        synthetic_spectrograms = await generator.generate_spectrograms(
            gan_model, num_samples
        )
        
        # Convert back to audio and save
        synthetic_dir = await generator.save_synthetic_audio(
            synthetic_spectrograms, agent_id, dataset_id
        )
        
        # Update dataset
        await db.update_dataset(dataset_id, {
            'synthetic_data_path': synthetic_dir,
            'synthetic_samples_count': len(synthetic_spectrograms),
            'gan_epochs': gan_epochs,
            'last_synthetic_generation': 'now'
        })
        
        # Log generation iteration
        await db.log_iteration({
            'agent_id': agent_id,
            'dataset_id': dataset_id,
            'type': 'synthetic_generation',
            'data_type': 'audio',
            'samples_generated': len(synthetic_spectrograms),
            'gan_epochs': gan_epochs,
            'status': 'completed'
        })
        
        logger.info(f"Audio GAN generation completed: {len(synthetic_spectrograms)} samples")
        
    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        await db.log_iteration({
            'agent_id': agent_id,
            'dataset_id': dataset_id,
            'type': 'synthetic_generation',
            'data_type': 'audio',
            'status': 'failed',
            'error': str(e)
        })

@router.get("/generation-status/{agent_id}")
async def get_generation_status(
    agent_id: str,
    db: JSONDatabase = Depends(get_db)
):
    """Get synthetic data generation status for agent"""
    try:
        iterations = await db.get_iterations(agent_id)
        
        # Filter synthetic generation iterations
        generation_iterations = [
            i for i in iterations 
            if i.get('type') == 'synthetic_generation'
        ]
        
        return {
            'agent_id': agent_id,
            'generation_history': generation_iterations,
            'total_generations': len(generation_iterations),
            'successful_generations': len([
                i for i in generation_iterations 
                if i.get('status') == 'completed'
            ])
        }
        
    except Exception as e:
        logger.error(f"Error getting generation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/synthetic-data-info/{dataset_id}")
async def get_synthetic_data_info(
    dataset_id: str,
    db: JSONDatabase = Depends(get_db)
):
    """Get information about synthetic data for a dataset"""
    try:
        datasets = await db.get_datasets()
        if dataset_id not in datasets:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset_info = datasets[dataset_id]
        
        synthetic_info = {
            'dataset_id': dataset_id,
            'has_synthetic_data': 'synthetic_data_path' in dataset_info,
            'synthetic_data_path': dataset_info.get('synthetic_data_path'),
            'synthetic_samples_count': dataset_info.get('synthetic_samples_count', 0),
            'last_generation': dataset_info.get('last_synthetic_generation'),
            'data_type': dataset_info.get('data_type'),
            'original_samples_count': dataset_info.get('samples_count', 0)
        }
        
        return synthetic_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting synthetic data info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/merge-synthetic-data/{dataset_id}")
async def merge_synthetic_with_original(
    dataset_id: str,
    merge_ratio: float = Form(0.3),  # 30% synthetic, 70% original
    db: JSONDatabase = Depends(get_db)
):
    """Merge synthetic data with original dataset"""
    try:
        datasets = await db.get_datasets()
        if dataset_id not in datasets:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset_info = datasets[dataset_id]
        
        if 'synthetic_data_path' not in dataset_info:
            raise HTTPException(status_code=400, detail="No synthetic data available")
        
        # Determine data type and merge accordingly
        data_type = dataset_info.get('data_type', 'text')
        
        if data_type == 'text':
            generator = TextSyntheticGenerator()
            merged_path = await generator.merge_datasets(
                dataset_info['file_path'],
                dataset_info['synthetic_data_path'],
                merge_ratio
            )
        elif data_type == 'image':
            generator = ImageGANGenerator()
            merged_path = await generator.merge_image_datasets(
                dataset_info['file_path'],
                dataset_info['synthetic_data_path'],
                merge_ratio
            )
        elif data_type == 'audio':
            generator = AudioGANGenerator()
            merged_path = await generator.merge_audio_datasets(
                dataset_info['file_path'],
                dataset_info['synthetic_data_path'],
                merge_ratio
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported data type: {data_type}")
        
        # Update dataset info
        await db.update_dataset(dataset_id, {
            'merged_data_path': merged_path,
            'merge_ratio': merge_ratio,
            'last_merge': 'now'
        })
        
        return {
            'dataset_id': dataset_id,
            'merged_data_path': merged_path,
            'merge_ratio': merge_ratio,
            'status': 'success',
            'message': 'Datasets merged successfully'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error merging datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))