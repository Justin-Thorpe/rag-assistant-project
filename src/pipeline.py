from dataclasses import dataclass

from src import config
from src.integration import load_documents, split_documents
from src.retrieval import get_vector_store, index_documents
from src.generation import build_chain, format_context, format_sources

@dataclass
class Answer:
    text: str
    sources: list[dict]

class RagPipeline():
    def __init__(self):
        self.vectorstore = get_vector_store()
        if self.vectorstore._collection.count() == 0:
            print("No vector store found. Indexing documents...")
            splits = split_documents(load_documents(config.RAW_DATA_DIR))
            index_documents(self.vectorstore, splits)
            print(f"Indexed {len(splits)} chunks.")
        self.chain = build_chain()

    def ask(self, question):
        results = self.vectorstore.similarity_search_with_relevance_scores(question, k=config.TOP_K)
        docs = [d for d, score in results if score >= config.MIN_RELEVANCE]

        if not docs:
            return Answer("I could not find that information in the indexed documents.", [])
        
        response = self.chain.invoke({"context": format_context(docs), "question": question})

        return Answer(response.content, format_sources(docs))