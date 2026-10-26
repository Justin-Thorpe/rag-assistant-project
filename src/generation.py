# Generation handles prompt templates, LLM setup, and answer generation

from src import config
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

SYSTEM_PROMPT = ChatPromptTemplate.from_messages(
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

def create_llm():
    return ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMP)

def build_chain():
    '''
    Creates a prompt and llm once, then returns those
    '''
    llm = create_llm()

    return SYSTEM_PROMPT | llm

def format_context(docs):
    return "\n\n".join(
        f"[{i}] {d.metadata['source']}, page {d.metadata['page']}\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )

def format_sources(docs):
    '''
    One entry per (source, page), with chunk numbers the LLM can return
    '''
    sources = {}
    for i, d in enumerate(docs, start=1):
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key not in sources:
            sources[key] = {
                "source": key[0],
                "page": key[1],
                "refs": [],
                "excerpt": d.page_content[:300]
            }
        sources[key]["refs"].append(i)

    return list(sources.values())