ml_backend/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI main application
│   ├── config.py                  # Configuration settings
│   ├── database/
│   │   ├── __init__.py
│   │   ├── json_db.py            # JSON-based database manager
│   │   └── models.py             # Pydantic models
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── training.py       # Training endpoints
│   │   │   ├── explanation.py    # SHAP explanation endpoints
│   │   │   ├── synthetic_data.py # Synthetic data generation
│   │   │   ├── questionnaire.py  # Questionnaire generation
│   │   │   ├── agents.py         # Domain agent endpoints
│   │   │   ├── datasets.py       # Dataset management
│   │   │   └── vectordb.py       # Vector DB operations
│   ├── core/
│   │   ├── __init__.py
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── ml_trainer.py     # ML model training
│   │   │   ├── dl_trainer.py     # Deep learning training
│   │   │   └── model_factory.py  # Model creation factory
│   │   ├── explanation/
│   │   │   ├── __init__.py
│   │   │   └── shap_explainer.py # SHAP explanations
│   │   ├── synthetic_data/
│   │   │   ├── __init__.py
│   │   │   ├── text_generator.py # Text synthetic data
│   │   │   ├── image_gan.py      # Image GAN generation
│   │   │   └── audio_gan.py      # Audio/Spectrogram GAN
│   │   ├── langraph/
│   │   │   ├── __init__.py
│   │   │   ├── coordinator.py    # LangGraph coordination
│   │   │   └── agents.py         # Domain-specific agents
│   │   └── vectordb/
│   │       ├── __init__.py
│   │       └── qdrant_manager.py # Qdrant operations
│   ├── utils/
│   │   ├── __init__.py 
│   │   ├── data_processing.py    # Data preprocessing utilities
│   │   ├── chunk_manager.py      # Chunk management
│   │   └── logging.py            # Logging utilities
│   └── schemas/
│       ├── __init__.py
│       ├── training.py           # Training request/response schemas
│       ├── explanation.py        # Explanation schemas
│       └── synthetic_data.py     # Synthetic data schemas
├── data/
│   ├── datasets/                 # Dataset storage
│   │   ├── text/
│   │   ├── images/
│   │   └── audio/
│   ├── models/                   # Trained model storage
│   ├── chunks/                   # Chunk storage
│   └── database/
│       ├── agents.json           # Agent configurations
│       ├── models.json           # Model metadata
│       ├── datasets.json         # Dataset metadata
│       └── iterations.json       # Training iterations log
├── logs/                         # Application logs
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── README.md   