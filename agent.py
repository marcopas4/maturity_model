from pathlib import Path
import time
import config as cfg
from llama_index.core import Settings
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from groq import Groq
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

class Agent:
    """
    Agent is the object that contains the state of the conversation
    """
    def __init__(self):
        self.curr_question: str = None  # qestionnaire question
        self.question_id: int = 0# question ID

        
        self.answer = None # Answer generated
        self.context: str =None  #  retrieved documents
        self.questions: List[str] = []
        self.llm = Groq(api_key=cfg.GROQ_API_KEY)
        self.retriever = None
        self.nodes = None
        self.docstore = None
    
        """
        Nodo di inizializzazione:
        1. Carica il questionario da un dataframe
        2. Inizializza i documenti per il retrieval RAG
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
            chunk_sizes=[512,8192],
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
    
    def retrieve(self,query:str,mode:str) -> List[Node]:
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
        
    def grade_context(self,context) -> bool:
        """
        Evaluate if the context is relevant for answering the current question.
        
        Returns:
            bool: True if context is relevant, False otherwise
        
        Raises:
            RuntimeError: If LLM completion fails
            json.JSONDecodeError: If response is not valid JSON
            ValueError: If response value is not boolean
        """
        print("---GRADE CONTEXT---")
        time.sleep(2)
        
        try:
            completion = self.llm.chat.completions.create(
                model=cfg.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Sei un assistente. Valuta se il contenuto di questo contesto è utile anche in minima parte per rispondere alla domanda. Rispondi SOLO in formato JSON contenente una chiave 'response' con valore booleano true o false, esempio: {'response': true}"
                    },
                    {
                        "role": "user",
                        "content": f"Domanda: {self.curr_question}. Contesto: {context}"
                    },
                ],
                temperature=0.6,
                top_p=0.95,
                stream=False,
                response_format={"type": "json_object"},
                stop=None,
            )

            # Parsa la risposta JSON
            response = completion.choices[0].message.content
            response_dict = json.loads(response)
            
            # Verifica che 'response' contenga un booleano
            if not isinstance(response_dict.get('response'), bool):
                raise ValueError("La risposta deve essere un booleano (true/false)")
                
            return response_dict['response']
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Errore nel parsing JSON: {str(e)}")
        except KeyError:
            raise ValueError("Il JSON non contiene la chiave 'response'")
        

    def query_expansion(self,query:str) -> str:
        """
        Expands the current question using the LLM.
        
        Returns:
            str: Expanded question
        
        Raises:
            RuntimeError: If LLM completion fails
            json.JSONDecodeError: If response is not valid JSON
        """
        print("---QUERY EXPANSION---")
        time.sleep(2)
        
        try:
            completion = self.llm.chat.completions.create(
                model=cfg.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Sei un assistente.Ti verrà data in input una query. Riformula la query mantenendo il significato originale ma usando altri termini. Rispondi SOLO in formato JSON contenente una chiave 'response' con valore stringa, esempio: {'response': 'query espansa'}"
                    },
                    {
                        "role": "user",
                        "content": f"Domanda: {query}"
                    },
                ],
                temperature=0.6,
                top_p=0.95,
                stream=False,
                response_format={"type": "json_object"},
                stop=None,
            )
            
            # Parsa la risposta JSON
            response = completion.choices[0].message.content
            
                
            return response
            
        except Exception as e:
            raise ValueError(f"Errore nella query expansion: {e}")
    
    def generate_answer(self):
        print("---GENERATE ANSWER---")
        time.sleep(2)
        completion = self.llm.chat.completions.create(
            model=cfg.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Rispondi alla domanda in formato JSON usando questo schema : {response : str} , dove str è una stringa compresa tra queste risposte standard, utilizzando il contesto fornito. Le stringhe standard sono : 1) Si ; 2)Si ,ma senza una struttura ben definita; 3)No."
                },
                {
                    "role": "user",
                    "content": f"Domanda: {self.curr_question}. Contesto: {self.context}"
                },
            ],
            temperature=0.6,
            top_p=0.95,
            stream=False,
            response_format={"type": "json_object"},
            stop=None,
        )
        self.answer = completion.choices[0].message.content


    def compile_and_save(self,answer:str):
        """
        Save current question and answer to a CSV file.
        If the file doesn't exist, it will be created with headers.
        """
        print("---COMPILE AND SAVE---")
        
        
        
        # Define the CSV file path
        csv_file = 'results/responses.csv'
        os.makedirs('output', exist_ok=True)
        
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









