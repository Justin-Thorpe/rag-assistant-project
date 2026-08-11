# Retrieval handles embedding creation, vectore setup, and retriever configuration

from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def build_vector_store(splits):
    embeddings = create_embeddings()

    return Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="f1_regulations",
        persist_directory=str(config.VECTOR_STORE_DIR),
    )

def create_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_kwargs={"k": config.TOP_K}
    )

def retrieve_documents(question, retriever):
    return retriever.invoke(question)