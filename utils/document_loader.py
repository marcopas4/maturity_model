from pathlib import Path
from typing import List
from langchain_community.document_loaders import DirectoryLoader,PyMuPDFLoader
from langchain_core.documents import Document
import config as cfg

def load_documents() -> List[Document]:
    """
    Carica i documenti da una directory.
    
    Args:
        directory_path: Percorso della directory contenente i documenti.
        
    Returns:
        Lista di documenti caricati.
    """
    pdf_dir = cfg.DOCUMENTS_DIR

    try:
        loader = DirectoryLoader(
            str(pdf_dir),
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader
        )
        documents = loader.load()
        print(f"Caricati {len(documents)} documenti da {pdf_dir}")
        return documents
    except Exception as e:
        print(f"Errore durante il caricamento dei documenti: {e}")
        return []

