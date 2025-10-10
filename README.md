# Maturity Model Assessment Tool

## Overview
A comprehensive Python-based system for automated assessment of organizational information governance maturity levels using advanced RAG (Retrieval Augmented Generation), LLM technologies, and human-in-the-loop interaction. The tool supports both console and web-based GUI interfaces for flexible assessment workflows.

## 🚀 Features
- **Dual Interface Support**: Console-based and modern web GUI using Gradio
- **Human-in-the-Loop Processing**: Interactive context selection for improved accuracy
- **Advanced Document Analysis**: Multi-source PDF processing and intelligent retrieval
- **Smart Query Expansion**: Automatic query enhancement for comprehensive search
- **Context Reranking**: Jina AI-powered document reranking for relevance optimization
- **Real-time Processing**: Live feedback and status updates during assessment
- **Structured Output**: Excel-based results with comprehensive tracking
- **Answer Management**: Update and override capabilities for iterative refinement
- **Multi-threaded Processing**: Efficient document loading and processing

## 🏗️ Project Structure
```
maturity_model/
├── main.py                  # Main entry point with interface selection
├── agent.py                 # Core agent class for document processing & LLM interaction
├── compile.py              # Batch processing logic for console mode
├── gui.py                  # Gradio-based web interface
├── config.py               # Configuration settings and API keys
├── retriever.py           # Document retrieval system
├── metadata_retriever.py  # Metadata-based retrieval
├── jina_reranker.py       # Jina AI reranking implementation
├── utils/
│   ├── document_loader.py # Advanced PDF loading utilities
│   └── questionGen.py     # Question generation and loading
├── structured_classes.py  # Pydantic models for data validation
├── results/               # Generated assessment results
│   ├── responses.xlsx     # Main results file
│   └── logs/             # System logs
└── data/                 # Input documents and questionnaires
```

## 📋 Requirements
- **Python**: 3.10+
- **Key Dependencies**:
  - `gradio>=5.49.1` - Web interface
  - `llama-index>=0.14.2` - RAG framework
  - `openai>=1.108.2` - LLM integration
  - `pandas>=2.2.3` - Data processing
  - `openpyxl>=3.1.5` - Excel file handling
  - `sentence-transformers>=5.1.1` - Embedding models
  - `faiss-cpu>=1.12.0` - Vector search

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/maturity_model.git
cd maturity_model
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file or set environment variables:
```bash
# Required API Keys
export CEREBRAS_API_KEY=your_cerebras_api_key_here
export JINA_API_KEY=your_jina_api_key_here  # Optional for reranking
export OPENAI_API_KEY=your_openai_api_key_here  # Fallback option
```

## 🚀 Usage

### Quick Start
```bash
python main.py
```

### Interface Selection
When running the tool, choose your preferred interface:
- **[1] GUI (Gradio)**: Modern web-based interface with real-time interaction
- **[2] Console**: Traditional command-line batch processing

### GUI Mode Features
1. **Interactive Assessment**: 
   - View questions with progress tracking
   - Load and review relevant document contexts
   - Select optimal contexts for answer generation
   - Real-time AI response generation

2. **Context Management**:
   - Browse multiple retrieved contexts
   - Choose specific contexts or combine all
   - Skip questions when appropriate
   - Update answers through re-selection

3. **Live Results**: 
   - Immediate feedback on answer generation
   - Status updates throughout the process
   - Progress tracking across all questions

### Console Mode
- Automated batch processing of all questions
- Human context selection via CLI prompts
- Comprehensive logging and error handling
- Suitable for large-scale assessments

## ⚙️ Configuration

### Key Settings (`config.py`)
```python
# LLM Configuration
CEREBRAS_API_KEY = "your_api_key"
LLM_MODEL = "llama3.1-8b"

# Processing Settings
BATCH_SIZE = 10
MAX_RETRIEVAL_ATTEMPTS = 3
TOP_K_CONTEXTS = 5

# File Paths
RESULTS_DIR = "./results"
DATA_DIR = "./data"
RESPONSES_PATH = "./results/responses.xlsx"
```

### Advanced Features
- **Reranking**: Enable Jina AI reranker for improved context relevance
- **Query Expansion**: Automatic generation of related queries
- **Deduplication**: Smart removal of similar contexts
- **Logging**: Comprehensive system and user interaction logging

## 📊 Output Format

### Excel Results (`results/responses.xlsx`)
| Column | Description |
|--------|-------------|
| Question ID | Unique identifier for each question |
| Question | Full text of the assessment question |
| Answer | Generated response (Yes/No/Conditional/Skipped) |
| Context | Selected document context used for answer |
| Justification | AI-generated explanation for the answer |
| Timestamp | When the answer was generated/updated |

### Supported Answer Types
- **"Si"**: Requirement is fully met
- **"No"**: Requirement is not met
- **"Si, ma senza una struttura ben definita"**: Partially implemented
- **"SKIPPED"**: Question skipped by user
- **"Error"**: Processing error occurred

## 🔧 Architecture

### 1. Document Processing Pipeline
```
PDF Documents → Text Extraction → Chunking → Embedding → Vector Store
```

### 2. Question Processing Workflow
```
Question → Query Expansion → Document Retrieval → Reranking → Context Selection → Answer Generation
```

### 3. Human-in-the-Loop Integration
- **Context Review**: Users evaluate retrieved document contexts
- **Selection Interface**: Choose most relevant context for answer generation  
- **Iterative Refinement**: Update answers based on different context selections
- **Quality Control**: Human oversight ensures answer accuracy

### 4. Web Interface Architecture
- **Gradio Framework**: Modern, responsive web interface
- **Real-time Updates**: Live status and progress feedback
- **State Management**: Persistent session state across interactions
- **Error Handling**: Graceful error recovery and user notification

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make changes and add tests
5. Commit changes: `git commit -am 'Add new feature'`
6. Push branch: `git push origin feature/new-feature`
7. Open pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 🐛 Troubleshooting

### Common Issues
1. **API Key Errors**: Ensure all required API keys are properly set
2. **Memory Issues**: Reduce batch size for large document collections
3. **GUI Port Conflicts**: Change port in `launch_gui()` if 7860 is busy
4. **Excel File Locks**: Close Excel files before running assessments

### Logging
System logs are available in `results/logs/maturity_model.log` with detailed execution information.

## 📄 License
MIT License - See LICENSE file for details

## 👤 Author
**Marco Pasca**  
Master's Thesis Project  
University of Padova  
📧 marcopasca4@gmail.com

## 🙏 Acknowledgments
- **Cerebras Systems** for high-performance LLM API access
- **Jina AI** for advanced document reranking capabilities  
- **LlamaIndex Team** for the comprehensive RAG framework
- **Gradio Team** for the intuitive web interface framework
- **University of Padova** for academic support and guidance

## 🔗 Links
- [Project Repository](https://github.com/yourusername/maturity_model)
- [Documentation](https://github.com/yourusername/maturity_model/wiki)
- [Issue Tracker](https://github.com/yourusername/maturity_model/issues)

---
*For technical support or questions about implementation, please contact marcopasca4@gmail.com or open an issue in the repository.*
