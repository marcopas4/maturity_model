from pathlib import Path
from typing import List
from llama_index.readers.file import PyMuPDFReader
import config as cfg
from llama_index.core import Document
from docling.document_converter import DocumentConverter



def load_documents_pymupdf(directory: Path) -> list:
    """Carica tutti i documenti PDF dalla directory specificata."""
    documents = []
    reader = PyMuPDFReader()
    
    # Itera su tutti i file PDF nella directory
    for file_path in directory.glob("*.pdf"):
        try:
            docs = reader.load_data(file_path)
            doc_text = "\n\n".join([d.get_content() for d in docs])
            doc_text = Document(text=doc_text)
            documents.append(doc_text)
            print(f"Caricato: {file_path.name}")
        except Exception as e:
            print(f"Errore nel caricamento di {file_path.name}: {e}")
    
    return documents


def load_documents_docling(directory: Path) -> List[Document]:
    """
    Load and convert all PDF documents from specified directory using docling.
    
    Args:
        directory (Path): Path to directory containing PDF files
        
    Returns:
        List[Document]: List of converted documents
        
    Raises:
        Exception: If document loading or conversion fails
    """
    documents = []
    converter = DocumentConverter()
    
    # Iterate through all PDF files in directory
    for file_path in directory.glob("*.pdf"):
        try:
            # Convert PDF to text using docling
            doc_text = converter.convert(file_path)
            docling_text = doc_text.document.export_to_markdown()
            
            # Create Document object
            document = Document(text=docling_text)
            documents.append(document)
            
            print(f"Loaded: {file_path.name}")
            
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
    
    return documents