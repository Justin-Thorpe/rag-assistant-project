from src import config
from src.retrieval import get_vector_store
from src.generation import build_chain, format_sources

class Answer:
    text: str
    sources: list[dict]

class RagPipeline():
    def __init__(self):
        self.vectorstore = get_vector_store()
        self.chain = build_chain()

    def ask_question(self, question):
        results = self.vectorstore.similarity_search_by_image_with_relevance_score(question, k=config.TOP_K)
        docs = [d for d, score in results if score >= config.MIN_RELEVANCE]

        if not docs:
            return Answer("I could not find that information in the indexed documents.", [])
        
        response = self.chain.invoke({"context": format_sources(docs), "question": question})

        return Answer(response.content, format_sources(docs))