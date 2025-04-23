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
import structured_classes as sc
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
        self.curr_question: str = None  # qestionnaire question
        self.question_id: int = 0# question ID

        self.questions: List[str] = []
        self.llm = client = OpenAI(
                api_key=os.environ.get("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1"  # URL base for Groq
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
            docs = dl.load_documents_docling(cfg.DOCUMENTS_DIR)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                

                # Configura l'embedding model
            embed_model = HuggingFaceEmbedding(
                model_name="all-MiniLM-L6-v2",
                device=device
            )
            Settings.embed_model = embed_model 
        
            
            
            node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[256,4096],
            chunk_overlap=100, 
                )
            
            self.nodes = node_parser.get_nodes_from_documents(docs)

            
            self.docstore = SimpleDocumentStore()

            # insert nodes into docstore
            self.docstore.add_documents(self.nodes)

            self.retriever = rt.Retriever(self.docstore, self.nodes)
                
            
        except Exception as e:
            print(f"Errore durante l'inizializzazione del'Agente: {e}")
            exit(1)

        
    def format_docs(self,docs):
            return "\n\n".join(doc.get_content() for doc in docs)
    
    def retrieve(self, query: str, mode: str) -> List[Node]:
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
        print(f"---RETRIEVE MODE---")
        self.curr_question = query

        if mode == "auto-merging":
                retrieved_docs = self.retriever.retrieve(query,mode="auto-merging")
                return retrieved_docs
        elif mode == "metadata":
                retrieved_docs = self.retriever.retrieve(query,mode="metadata")
                return retrieved_docs
        elif mode == "sparse":
                retrieved_docs = self.retriever.retrieve(query,mode="sparse")
                return retrieved_docs
        else:
                raise ValueError("Invalid mode. Choose 'auto-merging', 'metadata', or 'sparse'.")
        
    def grade_context(self, query: str, context: str) -> bool:
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
        time.sleep(2)
        
        function_schema = {
            "name": "evaluate_context_relevance",
            "description": "Valuta se il contesto fornito è rilevante per rispondere alla query",
            "parameters": sc.RelevanceResponse.model_json_schema()
        }
        
        # Preparazione del prompt
        system_prompt = """
        Sei un assistente specializzato nella valutazione della rilevanza dei contenuti.
        Il tuo compito è determinare se il contesto fornito contiene informazioni pertinenti 
        che potrebbero aiutare a rispondere alla query dell'utente.
        
        Rispondi SOLO con true se il contesto è anche minimamente rilevante per la query,
        oppure false se il contesto è completamente irrilevante o non aiuterebbe in alcun modo
        a rispondere alla query.
        """
        
        try:
            # Chiamata all'API con function calling
            response = self.llm.chat.completions.create(
                model=cfg.LLM_TEST,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\n\nContesto: {context}"}
                ],
                tools=[{"type": "function", "function": function_schema}],
                tool_choice={"type": "function", "function": {"name": "evaluate_context_relevance"}}
            )
            
            # Estrazione e validazione del risultato
            function_call = response.choices[0].message.tool_calls[0].function
            function_args = function_call.arguments
            
            # Validazione con Pydantic
            result = sc.RelevanceResponse.model_validate_json(function_args)
            
            return result.is_relevant
        except Exception as e:
            raise RuntimeError(f"Errore durante la valutazione della rilevanza: {str(e)}")
        

    def query_expansion(self, query: str) -> str:
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
        time.sleep(2)
        
        try:
            
            function_schema = {
            "name": "generate_expanded_queries",
            "description": "Genera query espanse e parole chiave da una query originale",
            "parameters": sc.ExpandedQuery.model_json_schema()
            }
    
            # Preparazione del prompt
            system_prompt = """
            Sei un assistente esperto in information retrieval. Il tuo compito è espandere la query dell'utente 
            in modo da migliorare il recupero dei documenti rilevanti. Genera una versione alternativa della query. 
            Restituisci sempre e solo un output strutturato JSON valido.
            """
            
            # Chiamata all'API con la function call
            response = self.llm.chat.completions.create(
                model=cfg.LLM_TEST,  # o il modello Groq che stai utilizzando
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Espandi questa query: {query}"}
                ],
                tools=[{"type": "function", "function": function_schema}],
                tool_choice={"type": "function", "function": {"name": "generate_expanded_queries"}}
            )
            
            # Estrazione del contenuto della function call
            function_call = response.choices[0].message.tool_calls[0].function
            function_args = function_call.arguments
            
            # Parsing del JSON e validazione con Pydantic
            expanded_query = sc.ExpandedQuery.model_validate_json(function_args)
            response = expanded_query.expanded_query
            return response
            
        except Exception as e:
            raise ValueError(f"Errore nella query expansion: {e}")
    
    def generate_answer(self, query: str, context: str) -> sc.PredefinedAnswer:
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
        time.sleep(2)
            # Definizione dello schema della funzione per l'API OpenAI
        function_schema = {
            "name": "select_predefined_response",
            "description": "Seleziona una delle risposte predefinite basata sulla query dell'utente",
            "parameters": sc.ResponseGenerator.model_json_schema()
        }
        
        # Preparazione del prompt
        system_prompt = """
        Sei un assistente che deve selezionare una risposta tra tre opzioni predefinite:
        1. "Si"
        2. "No" 
        3. "Si, ma senza una struttura ben definita"
        
        Analizza attentamente la query dell'utente e seleziona l'opzione più appropriata in base al contesto fornito.
        Non inventare altre risposte o formati. Devi restituire SOLO una delle tre opzioni specificate.
        """
        
        # Chiamata all'API con la function call
        response = self.llm.chat.completions.create(
            model=cfg.LLM_TEST,  # o il modello Groq che stai utilizzando
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\n Context: {context}"}
            ],
            tools=[{"type": "function", "function": function_schema}],
            tool_choice={"type": "function", "function": {"name": "select_predefined_response"}}
        )
        
        # Estrazione del contenuto della function call
        function_call = response.choices[0].message.tool_calls[0].function
        function_args = function_call.arguments
        
        # Parsing del JSON e validazione con Pydantic
        result = sc.ResponseGenerator.model_validate_json(function_args)
        print(f"Selected response: {result.selected_response}")
        print(f"Justification: {result.justification}")
        return result.selected_response
        


    def compile_and_save(self, answer: str):
        """
        Saves Q&A pair to CSV file.
        
        Args:
            answer (str): Generated answer
            
        Creates CSV with columns:
            - Question ID
            - Question
            - Answer
            - Timestamp
            
        File location: cfg.RESULTS_DIR/responses.csv
        """
        print("---COMPILE AND SAVE---")
        
        
        
        # Define the CSV file path
        csv_file = os.path.join(cfg.RESULTS_DIR, 'responses.csv')
        os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
        
        # Check if file exists to determine if headers need to be written
        file_exists = os.path.isfile(csv_file)
        
        # Open file in append mode
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write headers if file is new
            if not file_exists:
                writer.writerow(['Question ID', 'Question', 'Answer', 'Timestamp'])
            
            # Write the current Q&A pair
            writer.writerow([
                self.question_id,
                self.curr_question,
                answer,
                time.strftime('%Y-%m-%d %H:%M:%S')
            ])
        
        print(f"Saved Q&A to {csv_file}")









