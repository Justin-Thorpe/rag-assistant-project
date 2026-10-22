# Generation handles prompt templates, LLM setup, and answer generation

from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

def create_llm():
    return ChatOllama(
    model="llama3.2",
    temperature=0
)

def format_context(docs):
    return "\n\n".join(
        f"[{i}] {d.metadata['source']}, page {d.metadate['page']}\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )

def generate_answer(question, documents):
    llm = create_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an assistant that answers questions about uploaded documents.

                Use only facts stated in the context. 

                If the answer is in the context:
                Return your answer and cite the passages you used with their bracket numbers, e.g. [2].

                If the answer is not in the context, say exactly:
                "I could not find that information in the indexed documents."

                Context:
                {context}
                """,
            ),
            ("human", "{question}"),
        ]
    )

    rag_chain = prompt | llm

    response = rag_chain.invoke(
        {
            "context": format_context(documents),
            "question": question,
        }
    )

    return response.content

def format_sources(documents):
    return [
        {
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
        }
        for doc in documents
    ]
