# Maturity Model Project

## Overview
A Python-based system for automated assessment of information governance maturity levels using RAG (Retrieval Augmented Generation) and LLM technologies.

## Features
- Document analysis and retrieval
- Query expansion for improved search
- Context-aware answer generation 
- Structured output format
- Multi-threaded document processing
- CSV result storage

## Project Structure
```
maturity_model/
├── agent.py                  # Main agent class for processing
├── compile.py               # Core processing logic
├── config.py               # Configuration settings
├── retriever.py           # Retriever class
├── metadata_retriever.py  # Metadata retriever class
├── utils/
│   ├── document_loader.py # PDF loading utilities
│   └── questionGen.py     # Question loading
├── structured_classes.py  # Pydantic models
├── results/              # Generated results
└── data/                # Input documents
```

## Requirements
- Python 3.9+
- Dependencies:
```
llama-index==0.9.8
openai==1.12.0
pydantic==2.6.1
docling==0.5.0
torch>=2.0.0
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/maturity_model.git
cd maturity_model
```

2. Create virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
set GROQ_API_KEY=your_api_key_here
```

## Usage
1. Place PDF documents in `data/` directory

2. Run the analysis:
```bash
python main.py
```

3. Results will be saved in `results/responses.csv`

## Configuration
Key settings in `config.py`:
- `LLM_MODEL`: LLM model selection
- `BATCH_SIZE`: Document processing batch size
- `MAX_ATTEMPTS`: Maximum retrieval attempts
- `RESULTS_DIR`: Output directory path

## Output Format
The system generates a CSV file with:
- Question ID
- Original question
- Generated answer
- Timestamp

## Architecture

### 1. Document Processing
- Multi-threaded PDF loading
- Text extraction and preprocessing
- Document store creation

### 2. Query Processing
- Question analysis
- Context retrieval
- Query expansion when needed

### 3. Answer Generation
- Context relevance assessment
- Structured answer generation
- Result validation

## Contributing
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Open pull request

## License
MIT License - See LICENSE file for details

## Author
Marco Pasca  
Master's Thesis Project  
University of Padova

## Acknowledgments
- Groq for API access
- LlamaIndex team
- Docling project

For more information, contact marcopasca4@gmail.com
