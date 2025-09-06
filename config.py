from pydantic_settings import BaseSettings # This is the correct way
import os

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ML Training Backend"
    
    # Database Settings
    DATABASE_PATH: str = "data/database"
    
    # Qdrant Settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str = ""
    
    # Groq Settings
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # Model Settings
    MAX_MODEL_SIZE: int = 500 * 1024 * 1024  # 500MB
    
    # Training Settings
    BATCH_SIZE: int = 32
    MAX_EPOCHS: int = 100
    LEARNING_RATE: float = 0.001
    
    # File Upload Settings
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: list = [".csv", ".json", ".jpg", ".png", ".wav", ".mp3"]
    
    # Synthetic Data Settings
    GAN_LATENT_DIM: int = 100
    GAN_EPOCHS: int = 1000
    
    # SHAP Settings
    SHAP_MAX_SAMPLES: int = 1000
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()