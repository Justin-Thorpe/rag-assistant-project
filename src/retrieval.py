# Retrieval handles embedding creation, vectore setup, and retriever configuration

from src import config
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL
    )

def build_vector_store(splits):
    embeddings = create_embeddings()

    return Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="f1_regulations",
        persist_directory=str(config.VECTOR_STORE_DIR),
    )

def get_vector_store():
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=create_embeddings(),
        persist_directory=str(config.VECTOR_STORE_DIR),
        collection_metadata={"hnsw:space":"cosine"},
    )

def index_documents(vectorstore, splits):
    ids = [doc.metadata["chunk_id"] for doc in splits]
    vectorstore.add_documents(splits, ids=ids)

def create_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_kwargs={"k": config.TOP_K}
    )

def retrieve_documents(question, retriever):
    return retriever.invoke(question)