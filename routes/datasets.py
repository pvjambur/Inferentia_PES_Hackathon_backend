from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import List, Dict, Any, Optional
import os
import uuid
import pandas as pd

# The main application file will provide this dependency.
# It should be defined in main.py to handle a single database instance.
from database.json_db import JSONDatabase
from database.models import DatasetMetadata
from utils.data_processing import DataProcessor
from utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

DATA_DIR = "./data/datasets"

# We'll use this dependency to get the database connection
def get_db_conn() -> JSONDatabase:
    # This dependency will be replaced by the one defined in main.py
    # and passed to the router via the `dependencies` parameter.
    return JSONDatabase(os.path.join(DATA_DIR, "metadata.json"))

@router.post("/datasets/", response_model=DatasetMetadata, status_code=201)
async def upload_dataset(
    file: UploadFile = File(...), 
    description: str = Form(...),
    db: JSONDatabase = Depends(get_db_conn)
):
    """
    Uploads a new dataset file, processes it, and stores its metadata.
    """
    dataset_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    dataset_path = os.path.join(DATA_DIR, f"{dataset_id}{file_extension}")

    # Ensure the data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        # Save the file to the unique path
        with open(dataset_path, "wb") as buffer:
            buffer.write(await file.read())

        # Process the dataset to get schema and sample count
        processor = DataProcessor()
        dataset_info = processor.process_dataset(dataset_path)

        new_dataset = DatasetMetadata(
            id=dataset_id,
            name=file.filename,
            description=description,
            file_path=dataset_path,
            file_type=file_extension,
            num_samples=dataset_info.get("num_samples"),
            data_schema=dataset_info.get("schema")
        )

        # Use the correct database method to create a record with the new dataset's ID
        db.create_record(dataset_id, new_dataset.model_dump())
        logger.info(f"Successfully uploaded and processed dataset: {file.filename} with ID {dataset_id}")
        return new_dataset
    except Exception as e:
        logger.error(f"Failed to upload dataset: {e}")
        # Clean up the file if an error occurs
        if os.path.exists(dataset_path):
            os.remove(dataset_path)
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {e}")

@router.get("/datasets/", response_model=List[DatasetMetadata])
async def get_all_datasets(db: JSONDatabase = Depends(get_db_conn)):
    """
    Retrieves metadata for all uploaded datasets.
    """
    try:
        # The `read_all` method now returns a list of values
        datasets_data = db.read_all()
        return [DatasetMetadata(**ds) for ds in datasets_data]
    except Exception as e:
        logger.error(f"Failed to retrieve datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve datasets: {e}")

@router.get("/datasets/{dataset_id}", response_model=DatasetMetadata)
async def get_dataset_metadata(dataset_id: str, db: JSONDatabase = Depends(get_db_conn)):
    """
    Retrieves metadata for a single dataset by its ID.
    """
    try:
        # The `read_record` method now takes the key directly
        dataset_data = db.read_record(dataset_id)
        if not dataset_data:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return DatasetMetadata(**dataset_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dataset: {e}")