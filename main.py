from fastapi import FastAPI, HTTPException, APIRouter, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any
import uuid
import json
# Internal imports from your project structure
from config import settings
from database.json_db import JSONDatabase
from core.vectordb.qdrant_manager import QdrantManager
from utils.logging import setup_logging
from routes import (
    training,
    explanation,
    synthetic_data,
    questionnaire,
    agents,
    datasets,
    vectordb
)
 # This is the correct way
# --- Groq Integration (New) ---
from groq import Groq
from dotenv import load_dotenv

# A simple wrapper for the Groq client to manage the API key and instantiation.
load_dotenv()

class GroqService:
    def __init__(self):
        # Retrieve the API key from the environment
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        self.client = Groq(api_key=api_key)

    def generate_content(self, prompt: str, model: str = "llama3-8b-8192") -> str:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Groq API error: {e}")
            raise HTTPException(status_code=500, detail=f"Groq API error: {e}")

# --- Setup and Lifespan ---

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting ML Backend Application...")
    
    # Initialize directories
    os.makedirs("data/datasets/text", exist_ok=True)
    os.makedirs("data/datasets/images", exist_ok=True)
    os.makedirs("data/datasets/audio", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/chunks", exist_ok=True)
    os.makedirs("data/database", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Initialize database
    db = JSONDatabase(file_path="data/training_history.json") 
    #await db.initialize()

    # This part is already correct, as it provides a file path.
    app.state.agents_db = JSONDatabase(file_path="data/agents.json")
    
    # Initialize Qdrant
    qdrant = QdrantManager()
    await qdrant.initialize()
    app.state.qdrant = qdrant
    
    # Initialize Groq client
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        logger.warning("GROQ_API_KEY not found in environment variables. Groq endpoints will be disabled.")
    else:
        app.state.groq = GroqService()
        
    logger.info("Application startup complete")
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    if hasattr(app.state, 'qdrant'):
        await app.state.qdrant.close()

# --- FastAPI App Initialization ---

app = FastAPI(
    title="ML Training & Explanation Backend",
    description="Comprehensive backend for ML/DL training with SHAP explanations and synthetic data generation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="data"), name="static")

# --- New Groq Router ---

groq_router = APIRouter(prefix="/api/v1/groq", tags=["groq"])
router = APIRouter()

@router.get("/status")
async def get_status():
    """
    Returns the status of the backend to check if the connection is active.
    """
    return JSONResponse(content={"status": "online"}, status_code=200)

def get_db_for_routes() -> JSONDatabase:
    return app.state.agents_db

app.include_router(router, prefix="/api/v1")

@groq_router.post("/generate-optimized-questionnaire")
async def generate_optimized_questionnaire(
    request: Dict[str, Any] = Body(...)
):
    """
    Generates a questionnaire using Groq and optimizes it with Qdrant.
    """
    if not hasattr(app.state, 'groq'):
        raise HTTPException(status_code=503, detail="Groq service is not available.")
    
    user_prompt = request.get("prompt")
    if not user_prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    # 1. Use Groq to generate a base questionnaire
    base_questionnaire = app.state.groq.generate_content(
        prompt=f"Generate a detailed and effective questionnaire based on this topic: {user_prompt}"
    )

    # 2. Generate embedding for the questionnaire
    # This is a placeholder for a real embedding model.
    # In production, you would use an embedding model like `sentence-transformers`.
    embedding = [0.1] * 128  # Simulated embedding
    
    # 3. Store the generated questionnaire and its embedding in Qdrant
    questionnaire_id = str(uuid.uuid4())
    point = {
        "id": questionnaire_id,
        "vector": embedding,
        "payload": {
            "type": "questionnaire",
            "content": base_questionnaire,
            "source": "groq",
            "prompt": user_prompt
        }
    }
    await app.state.qdrant.add_points([point], collection_name="groq-generated-content")
    
    return {
        "status": "success",
        "questionnaire_id": questionnaire_id,
        "questionnaire_content": base_questionnaire,
        "message": "Questionnaire generated and indexed in Qdrant."
    }

@groq_router.post("/generate-optimized-synthetic-data")
async def generate_optimized_synthetic_data(
    request: Dict[str, Any] = Body(...)
):
    """
    Generates synthetic data from a questionnaire using Groq and links to Qdrant.
    """
    if not hasattr(app.state, 'groq'):
        raise HTTPException(status_code=503, detail="Groq service is not available.")

    questionnaire_content = request.get("questionnaire_content")
    if not questionnaire_content:
        raise HTTPException(status_code=400, detail="Questionnaire content is required.")

    # 1. Use Groq to generate synthetic data based on the questionnaire
    synthetic_data = app.state.groq.generate_content(
        prompt=f"Based on this questionnaire, generate realistic synthetic data in JSON format: {questionnaire_content}"
    )

    # 2. (Optional) Search Qdrant for similar past synthetic data
    # This demonstrates the optimization loop.
    # In a real scenario, you'd embed the questionnaire_content and search.
    # For this example, we'll just show the concept.
    search_results = await app.state.qdrant.search(
        vector=[0.5] * 128,  # Simulated query vector
        collection_name="groq-generated-content"
    )
    
    return {
        "status": "success",
        "generated_data": synthetic_data,
        "qdrant_context": search_results,
        "message": "Synthetic data generated and contextualized with Qdrant."
    }


# --- Include Routers ---
app.include_router(agents.router, prefix="/api/v1", dependencies=[Depends(get_db_for_routes)])
app.include_router(training.router, prefix="/api/v1/training", tags=["training"])
app.include_router(explanation.router, prefix="/api/v1/explanation", tags=["explanation"])
app.include_router(synthetic_data.router, prefix="/api/v1/synthetic", tags=["synthetic-data"])
app.include_router(questionnaire.router, prefix="/api/v1/questionnaire", tags=["questionnaire"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
app.include_router(datasets.router, prefix="/api/v1/datasets", tags=["datasets"])
app.include_router(vectordb.router, prefix="/api/v1/vectordb", tags=["vectordb"])
app.include_router(groq_router) # Include the new router

# --- Root and Health Endpoints ---

@app.get("/")
async def root():
    return {
        "message": "ML Training & Explanation Backend",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ml-backend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None
    )