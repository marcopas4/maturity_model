from pathlib import Path
import time
import config as cfg
from llama_index.core import Settings
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import torch
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
)
import json
from llama_index.core.schema import Node
from llama_index.core import SimpleDirectoryReader
import numpy as np
from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from utils.questionGen import QuestionGen
import utils.document_loader as dl
import retriever as rt
import csv
import os
from openai import OpenAI
import pandas as pd
from openpyxl.styles import Font
from jina_reranker import JinaReranker


class Agent:
    """
    Manages conversation state, document retrieval, and answer generation using LLMs.
    
    Attributes:
        curr_question (str): Current question being processed
        question_id (int): Unique identifier for current question
        questions (List[str]): List of questions to process
        llm (OpenAI): LLM client instance using Groq's API
        retriever (Retriever): Document retrieval instance
        nodes (List[Node]): Document nodes for retrieval
        docstore (SimpleDocumentStore): Document storage system
    """
    def __init__(self):
        self.question_id: int = 0# question ID

        self.questions: List[str] = []
        self.llm = OpenAI(
            api_key=cfg.CEREBRAS_API_KEY,
            base_url="https://api.cerebras.ai/v1",  # Modifica dell'URL base per Cerebras
            default_headers={"Content-Type": "application/json"}
            )
        self.retriever = None
        self.nodes = None
        self.docstore = None
    
        """
        Initialization node:
        1. Load questionnaire from dataframe
        2. Initialize documents for RAG retrieval
        """
        try:
            print("---INIT---")
            # Estrae le domande dal file Excel
            
            self.questions = QuestionGen().extract_all_questions()
            
            # Inizializza il retirever and vectordb
            docs = dl.load_documents_pymupdf(cfg.DOCUMENTS_DIR)

            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

            # Configura l'embedding model
            embed_model = HuggingFaceEmbedding(
                model_name="all-MiniLM-L6-v2",
                device=device
            )
            Settings.embed_model = embed_model 
        
            
            
            node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[cfg.CHUNK_SIZE_PARENT,cfg.CHUNK_SIZE_CHILD],
            chunk_overlap=cfg.CHUNK_OVERLAP, 
                )
            
            self.nodes = node_parser.get_nodes_from_documents(docs)

            
            self.docstore = SimpleDocumentStore()

            # insert nodes into docstore
            self.docstore.add_documents(self.nodes)

            self.retriever = rt.Retriever(self.docstore, self.nodes)
            # Instantiate JinaReranker (drop-in replacement for Cohere reranker)
            # You can customize model_name/device/use_fp16 via config if needed
            self.reranker = JinaReranker(
                model_name=getattr(cfg, "JINA_MODEL_NAME", None) or "jinaai/jina-reranker-v2-base-multilingual",
                device=None,  # let JinaReranker auto-detect cuda/mps/cpu
                use_fp16=getattr(cfg, "JINA_USE_FP16", True),
                batch_size=getattr(cfg, "JINA_BATCH_SIZE", 32),
                max_length=getattr(cfg, "JINA_MAX_LENGTH", 1024),
                cache_dir=getattr(cfg, "JINA_CACHE_DIR", "./models")
            )
                
            
        except Exception as e:
            print(f"Errore durante l'inizializzazione del'Agente: {e}")
            exit(1)

        
    def format_docs(self,docs):
            return "\n\n".join(doc.get_content() for doc in docs)
    
    def retrieve(self, query: str) -> List[str]:
        """
        Retrieves relevant documents using specified mode.
        
        Args:
            query (str): Search query
            mode (str): Retrieval mode ('auto-merging', 'metadata', or 'sparse')
            
        Returns:
            List[Node]: List of retrieved document nodes
            
        Raises:
            ValueError: If invalid mode specified
        """
        try:
            print(f"---RETRIEVE MODE---")
            contexts = []
            response = self.retriever.retrieve(query)
            for doc in response:
                contexts.append(doc.get_content())
            return contexts

        except ValueError as e:
            print(f"Errore durante il recupero dei documenti: {e}")
            raise ValueError(f"Invalid retrieval mode: {e}")
    
    def rerank(self, query: str, contexts: List[str]):
        """
        Reranks contexts based on their relevance to the query using Cohere's reranking model.
        
        Args:
            query (str): The user query.
            contexts (List[str]): List of context strings to rerank.
            top_k (int): Number of top contexts to return.
            
        Returns:
            List[dict]: List of dictionaries with context and its score, sorted by relevance.
        """
        try:
            print("---RERANK---")
            rerank_results = self.reranker.rerank(
            model='rerank-v3.5',
            query=query,
            documents=contexts
            )
    
            docs_reranked = [contexts[result.index] for result in rerank_results.results]

        
            return docs_reranked
        except Exception as e:
            print(f"Errore durante il reranking dei contesti: {e}")
            raise ValueError(f"Reranking failed: {e}")

        

    def grade_context(self,query: str, context: str) -> bool:
        """
        Evaluates context relevance for the query.
        
        Args:
            query (str): User query
            context (str): Retrieved document context
            
        Returns:
            bool: True if context is relevant
            
        Raises:
            RuntimeError: If LLM evaluation fails
        """
        print("---GRADE CONTEXT---")
        time.sleep(1)
        
    
        
        # Preparazione del prompt
        system_prompt = """
        Sei un esperto valutatore di pertinenza dei contenuti. Il tuo compito è valutare 
        attentamente se il contesto fornito contiene informazioni realmente utili e pertinenti 
        per rispondere alla query dell'utente.

        Criteri di valutazione:
        1. Specificità: Il contesto deve contenere informazioni specificamente correlate alla query
        2. Utilità: Le informazioni devono essere utili per formulare una risposta
        3. Cerca di capire se dal contenuto è possibile rispondere alla domanda anche in modo indiretto
   
        
        Devi fornire in formato JSON strutturato:
        1. Un giudizio booleano (is_relevant)
        2. Una spiegazione dettagliata del tuo ragionamento (reasoning)
        """
        
        try:
            # Chiamata all'API con function calling
            response = self.llm.chat.completions.create(
                model=cfg.LLM_LLAMA_70B, 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\n\nContesto: {context}"}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Estrazione e validazione del risultato
            result = json.loads(response.choices[0].message.content)
            if not isinstance(result, dict) or 'is_relevant' not in result or 'reasoning' not in result:
                raise ValueError("JSON di risposta non valido o incompleto")
            print(f"Risultato: {result['reasoning']}")
            return result['is_relevant']
        except Exception as e:
            raise RuntimeError(f"Errore durante la valutazione della rilevanza: {str(e)}")
        

    def query_expansion(self, query: str) -> List[str]:
        """
        Expands query using LLM for better retrieval.
        
        Args:
            query (str): Original query
            
        Returns:
            str: Expanded query
            
        Raises:
            ValueError: If expansion fails
        """
        print("---QUERY EXPANSION---")
        time.sleep(1)
        
        try:
            
    
            # Preparazione del prompt
            system_prompt = '''Sei un assistente intelligente specializzato nell'analisi e nella riformulazione di query.  
            Data una singola “query originale”, procedi nel modo seguente:

            1. Genera **tre sotto-query**:
            - Le prime **due** devono essere formulate come **domande**.
            - La terza deve essere una **query di estrazione dei concetti chiave** (keyword extraction).
            2. Restituisci il risultato **in puro JSON**, con questa struttura ridotta ai soli campi richiesti:

            Esempio:
            query_originale: "Quali sono i benefici della meditazione ?"

            ```json
            {
            "sotto_query": [
                {
                "id": 1,
                "subquery": "Quali sono gli effetti della meditazione sulla riduzione dello stress?"
                },
                {
                "id": 2,
                "subquery": "In che modo la meditazione migliora le funzioni cognitive?"
                },
                {
                "id": 3,
                "subquery": "benefici, meditazione, stress, funzioni cognitive"
                }
            ]
            }
            '''
            
            # Chiamata all'API con la function call
            response = self.llm.chat.completions.create(
                model=cfg.LLM_QWEN,  # o il modello Groq che stai utilizzando
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Espandi questa query: {query}"},
                ],
                temperature=cfg.LLM_TEMPERATURE,
                response_format={"type": "json_object"}

            )
            
            
            expanded_queries = json.loads(response.choices[0].message.content)
            sotto_query = expanded_queries.get('sotto_query', [])
            if not isinstance(sotto_query, list):
                raise ValueError("Campo 'sotto_query' mancante o non valido")
            response = [subquery['subquery'] for subquery in sotto_query if 'subquery' in subquery]
            return response
            
        except Exception as e:
            raise ValueError(f"Errore nella query expansion: {e}")
    
    def generate_answer(self, query: str, context: str) -> str:
        """
        Generates predefined answer based on query and context.
        
        Args:
            query (str): User query
            context (str): Retrieved context
            
        Returns:
            PredefinedAnswer: One of three possible responses:
                - "Si"
                - "No"
                - "Si, ma senza una struttura ben definita"
        """
        print("---GENERATE ANSWER---")
        time.sleep(1)
        
        # Preparazione del prompt
        system_prompt = """
        Sei un assistente che deve selezionare una risposta tra tre opzioni predefinite:
        1. "Si"
        2. "No" 
        3. "Si, ma senza una struttura ben definita"
        
        Analizza attentamente la query dell'utente e seleziona l'opzione più appropriata in base al contesto fornito.
        Non inventare altre risposte o formati. Devi restituire SOLO una delle tre opzioni specificate in formato JSON.
        La risposta deve essere strutturata come segue:
        
        ```json
        {
            "response": "Si",
            "justification": "La risposta è corretta perché..."
        }
        """
        try:
            # Chiamata all'API con la function call
            response = self.llm.chat.completions.create(
                model=cfg.LLM_LLAMA_70B,  
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\n Context: {context}"}
                ],
                temperature=cfg.LLM_TEMPERATURE,
                response_format={"type": "json_object"}
            )
            # Estrazione e validazione del risultato
            result = json.loads(response.choices[0].message.content)
            if not isinstance(result, dict) or 'response' not in result or 'justification' not in result:
                raise ValueError("JSON di risposta non valido o incompleto")
            return result['response'], result['justification']
        except Exception as e:
            raise ValueError(f"Errore durante la generazione della risposta: {e}")
        
        
        


    def compile_and_save(self,question:str, answer: str,context: str,justification: str):
        """
        Saves Q&A pair to an Excel file.
        
        Args:
            question_id (int): Question identifier
            question (str): Question text
            answer (str): Generated answer
                
        Creates Excel file with columns:
            - Question ID
            - Question
            - Answer
            - Context
            - justification
            - Timestamp
            
        File location: cfg.RESULTS_DIR/responses.xlsx
        """
        print("---COMPILE AND SAVE---")
        
        excel_file = os.path.join(cfg.RESULTS_DIR, 'responses.xlsx')
        os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
        
        new_data = pd.DataFrame({
            'Question ID': [self.question_id],
            'Question': [question],
            'Answer': [answer],
            'Context': [context],
            'Justification': [justification],
            'Timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')]
        })
        
        try:
            if os.path.isfile(excel_file):
                existing_df = pd.read_excel(excel_file)
                updated_df = pd.concat([existing_df, new_data], ignore_index=True)
            else:
                updated_df = new_data
                
            # Create a new Excel writer object
            writer = pd.ExcelWriter(excel_file, engine='openpyxl')
            
            # Write the DataFrame without index
            updated_df.to_excel(writer, sheet_name='Responses', index=False)
            
            # Access the worksheet
            workbook = writer.book
            worksheet = workbook['Responses']
            
            # Set column widths
            column_widths = {
                'A': 12,  # Question ID
                'B': 50,  # Question
                'C': 20,  # Answer
                'D': 50,  # Context
                'E': 40,  # Justification
                'F': 20   # Timestamp
            }
            
            for col, width in column_widths.items():
                worksheet.column_dimensions[col].width = width
            
            # Style header row
            header_font = Font(bold=True)
            for cell in worksheet[1]:
                cell.font = header_font
                
            # Save and close
            writer.close()
            
            print(f"Saved Q&A to {excel_file}")
            
        except Exception as e:
            print(f"Error while handling Excel file: {e}")
            raise
    
    def jaccard_similarity(self,text1: str, text2: str) -> float:
                """Calcola Jaccard similarity tra due testi"""
                set1 = set(text1.lower().split())
                set2 = set(text2.lower().split())
                intersection = set1.intersection(set2)
                union = set1.union(set2)
                return len(intersection) / len(union) if union else 0

    def deduplicate_post_reranking(self,ranked_contexts: List[str], 
                                similarity_threshold: float = 0.95) -> List[str]:
        """
        Rimuove duplicati da lista di stringhe già ordinata per score decrescente
        Mantiene il primo (score più alto) di ogni gruppo di duplicati
        """
        if not ranked_contexts:
            return []
        
        unique_contexts = []
        
        for context in ranked_contexts:
            is_duplicate = False
            
            for unique_ctx in unique_contexts:
                # Usa similarity testuale (Jaccard)
                similarity = self.jaccard_similarity(context, unique_ctx)
                if similarity > similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_contexts.append(context)
        
        return unique_contexts










