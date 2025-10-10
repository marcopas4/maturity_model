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
CHUNK_SIZE_PARENT = 4000
CHUNK_SIZE_CHILD = 256
CHUNK_OVERLAP = 100
RETRIEVER_K = 15

# Parametri LLM
LLM_QWEN = "qwen-3-32b"
LLM_LLAMA_70B = "llama-3.3-70b"
LLM_TEMPERATURE = 0.2
LLM_GPT_OSS = "gpt-oss-120b"
# OpenAI API (da impostare come variabile d'ambiente)
'''GROQ_API_KEY = os.getenv('GROQ_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')'''
CEREBRAS_API_KEY = os.getenv('CEREBRAS_API_KEY')

