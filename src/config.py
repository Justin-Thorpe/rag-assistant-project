# Config settings for project

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = Path(
    os.getenv("RAW_DATA_DIR", PROJECT_ROOT/"data"/"raw")
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_PROJECT = os.getenv(
    "LANGSMITH_PROJECT",
    "rag-project"
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4