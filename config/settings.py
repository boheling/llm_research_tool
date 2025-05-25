import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Database Settings
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/llm_research")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/llm_research")

# NVIDIA API Settings (Evo2)
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_API_BASE_URL = os.getenv("NVIDIA_API_BASE_URL", "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions")
EVO2_MODEL_ID = os.getenv("EVO2_MODEL_ID")

# Hugging Face API Settings (ESM-1b)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
ESM_MODEL_ID = "facebook/esm-1b"

# Model Parameters
MAX_DNA_LENGTH = int(os.getenv("MAX_DNA_LENGTH", "512"))
MAX_PROTEIN_LENGTH = int(os.getenv("MAX_PROTEIN_LENGTH", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

# Model Settings
DNA_MODEL_NAME = os.getenv("DNA_MODEL_NAME", "dna-bert")
PROTEIN_MODEL_NAME = os.getenv("PROTEIN_MODEL_NAME", "protein-bert")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./model_cache")

# Security Settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Logging Settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "app.log") 