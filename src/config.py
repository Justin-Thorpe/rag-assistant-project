# Config settings for project

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

COLLECTION_NAME = "f1_regulations"
LLM_MODEL = "llama3.2"
LLM_TEMP = 0.1
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Tune
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4
MIN_RELEVANCE = 0.3