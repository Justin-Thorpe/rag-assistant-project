# Integration handles functions for loading and splitting docs

from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load Documents method
def load_documents(file_path: str) -> list[Document]:
    """
    Loads Every PDF in a directory as page-level LangChain documents
    """
    documents = []

    for pdf_path in Path(file_path).glob("*pdf"):
        reader = PdfReader(pdf_path)

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text and text.strip():
                documents.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source":pdf_path.name,
                            "page": page_number
                        }
                    )
                )
    if not documents:
        raise ValueError(f"No readable PDFs found in {file_path}")

    return documents

# Split method
def split_documents(
        documents,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    ):
    """
    Splits documents into a split list of documents based on chunk parameters
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return text_splitter.split_documents(documents)