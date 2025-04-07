import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Percorsi
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
QUESTIONS_PATH = DATA_DIR / "OurMaturityModel_remapped_1.xlsx"
DOCUMENTS_DIR = DATA_DIR / "documents"
RESULTS_DIR = BASE_DIR / "results"
RESPONSES_PATH = RESULTS_DIR / "responses.csv"

# Parametri RAG
CHUNK_SIZE_PARENT = 8192
CHUNK_SIZE_CHILD = 512
CHUNK_OVERLAP = 100
RETRIEVER_K = 7

# Parametri LLM
LLM_MODEL = "gemma2-9b-it"
LLM_TEMPERATURE = 0.5

# OpenAI API (da impostare come variabile d'ambiente)
GROQ_API_KEY = os.getenv('GROQ_API_KEY')