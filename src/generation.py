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

def format_context(retrieved_docs):
    return "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

def generate_answer(question, documents):
    llm = create_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an assistant that answers questions about Formula 1 regulations.

                Answer only using the provided context, in this case the F1 Regulations.

                If the answer is not in the context, say:
                "I could not find that information in the indexed documents."

                If the answer is in the context:
                Return an answer and ensure that you include a reference to the regulation section.

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
