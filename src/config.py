# Config settings for project

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LANGSMITH_PROJECT = os.getenv(
    "LANGSMITH_PROJECT",
    "rag-project"
)

CHUNK_SIZE = None
CHUNK_OVERLAP = None
TOP_K = None