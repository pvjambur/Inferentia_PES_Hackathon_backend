from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any
import os
import numpy as np

# All imports are correct.
from core.vectordb.qdrant_manager import QdrantManager
from utils.chunk_manager import ChunkManager
from database.models import DatasetMetadata
from database.json_db import JSONDatabase
from utils.logging import get_logger

router = APIRouter()
qdrant_manager = QdrantManager()
chunk_manager = ChunkManager()
logger = get_logger(__name__)

# This dependency is the same as the one in the first file.
# It should be defined in your main.py to manage a single instance.
def get_db_conn() -> JSONDatabase:
    # This is a placeholder; the main application will provide the real instance.
    return JSONDatabase(os.path.join("data", "datasets", "metadata.json"))

@router.post("/vectordb/index-dataset/{dataset_id}", status_code=202)
async def index_dataset_to_vectordb(dataset_id: str, db: JSONDatabase = Depends(get_db_conn)):
    """
    Chunks and indexes a specified dataset into the vector database.
    """
    try:
        # The `db.read_record` method is the correct way to get a single item.
        dataset_meta_data = db.read_record(dataset_id)
        if not dataset_meta_data:
            raise HTTPException(status_code=404, detail="Dataset not found.")

        dataset = DatasetMetadata(**dataset_meta_data)
        
        # Check for supported file types.
        if dataset.file_type.lower() not in ['.txt', '.md', '.csv']:
            raise HTTPException(status_code=400, detail="Only text-based datasets can be indexed at this time.")

        chunks = chunk_manager.create_chunks_from_file(dataset.file_path)
        
        if not chunks:
            raise HTTPException(status_code=500, detail="Failed to create chunks from the dataset.")

        # Assume we have an embedding model to get vectors
        # For this example, we'll simulate the process
        points = [
            {
                "id": i,
                "vector": [0.1 * i] * 128,  # Simulated vector
                "payload": {"text": chunk, "dataset_id": dataset.id}
            }
            for i, chunk in enumerate(chunks)
        ]
        
        qdrant_manager.add_points(points, collection_name=f"dataset_{dataset.id}")

        logger.info(f"Successfully indexed dataset {dataset.id} into vector DB.")
        return {"status": "indexing started", "details": f"Dataset {dataset.id} is being indexed."}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to index dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to index dataset: {e}")

@router.get("/vectordb/search/{collection_name}", response_model=List[Dict[str, Any]])
async def search_vectordb(collection_name: str, query: str):
    """
    Performs a vector search in the specified collection.
    """
    try:
        # Simulate getting the query vector
        query_vector = [0.5] * 128  # Placeholder for a real embedding
        search_results = qdrant_manager.search(query_vector, collection_name)
        return search_results
    except Exception as e:
        logger.error(f"Failed to search vector DB: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search vector DB: {e}")