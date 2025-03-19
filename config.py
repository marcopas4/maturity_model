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
CHUNK_SIZE_PARENT = 2000
CHUNK_SIZE_CHILD = 400
CHUNK_OVERLAP = 100
RETRIEVER_K = 5

# Parametri LLM
LLM_MODEL = "deepseek-r1-distill-qwen-32b"
LLM_KEYWORD = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0

# OpenAI API (da impostare come variabile d'ambiente)
GROQ_API_KEY = os.getenv('GROQ_API_KEY')