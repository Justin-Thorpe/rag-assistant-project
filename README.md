# F1 Regulations RAG Assistant

A Retrieval-Augmented Generation (RAG) notebook for answering questions about the 2026 Formula 1 regulations.

The project includes document integration, text chunking, local embeddings, vector search, retrieval, and local LLM generation using a simple modular Python architecture.

## Features

* Loads Formula 1 regulation PDFs
* Splits documents into searchable chunks
* Generates local embeddings with Hugging Face
* Stores vectors in Chroma
* Retrieves the most relevant regulation sections
* Generates answers using a local Ollama model
* Returns document source and page metadata
* Runs locally, without paid API keys

## Architecture

```text
PDF Regulations
      ↓
Document Loading
      ↓
Text Splitting
      ↓
Hugging Face Embeddings
      ↓
Chroma Vector Store
      ↓
Similarity Retrieval
      ↓
Ollama LLM
      ↓
Answer + Sources
```

## Tech Stack

* Python
* LangChain
* Hugging Face Sentence Transformers
* Chroma
* Ollama
* PyPDF
* Jupyter Notebook

## Project Structure

```text
rag-project/
├── README.md
├── .gitignore
├── .env.example
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── rag_build.ipynb
│   └── rag_evaluate.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── integration.py
│   ├── retrieval.py
│   └── generation.py
└── vector_store/
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.\.venv\Scripts\activate
```

Install CPU PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

Install Ollama and download the local model:

```bash
ollama pull llama3.2
```

Place the Formula 1 regulations (found here: https://api.fia.com/regulation/category/110) in:

```text
data/raw/
```

Then open:

```text
notebooks/rag_evaluate.ipynb
```

and run the notebook from top to bottom.

## Example

```python
question = "What happens if a race is delayed?"

answer, sources = ask_question(question)

print(answer)
print(sources)
```

Example output:

```text
Answer:
The applicable procedure depends on the reason and timing of the delay...

Sources:
2026 Formula 1 Sporting Regulations.pdf, page 42
```

## Current Scope

This project is intentionally designed as a minimum viable RAG application.

The current implementation uses:

* A single Formula 1 regulation knowledge base
* Basic similarity retrieval
* A fixed number of retrieved chunks
* A local embedding model
* A local language model

Advanced RAG features such as query rewriting, reranking, multi-query retrieval, routing, and agentic workflows are outside the current MVP scope.

## Possible Enhancements

* FastAPI endpoint for question answering
* Retrieval accuracy metrics
* Improved source citation formatting
* Optional hosted LLM providers
* Reranking or hybrid retrieval
* Simple web interface

## Purpose

The goal of this project is to demonstrate practical understanding of AI integration and the core RAG system.

It shows how external documents can be transformed into embeddings, indexed in a vector database, retrieved based on semantic similarity, and provided as grounded context to a language model.

## Recent Updates

* Fixed vector store duplication
* Added TOML
* 
