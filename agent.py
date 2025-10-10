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
import numpy as np
from typing import List
from utils.questionGen import QuestionGen
import utils.document_loader as dl
import retriever as rt
import csv
import os
from openai import OpenAI
import pandas as pd
from openpyxl.styles import Font
from jina_reranker import JinaReranker
import logging
import traceback


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
            self.questions = QuestionGen().extract_all_questions()
            docs = dl.load_all_documents(cfg.DOCUMENTS_DIR)

            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

            # Configura l'embedding model con maggiore determinismo
            embed_model = HuggingFaceEmbedding(
                model_name="Qwen/Qwen3-Embedding-0.6B",
                device=device
                
            )
            
            # Set seeds per determinismo (se supportato)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
            
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
        """
        logger = logging.getLogger(__name__)
        max_retries = 3
        backoff = 1.0
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"RETRIEVE MODE - Attempt {attempt}/{max_retries} - Query: {query[:100]}...")
                contexts = []
                response = self.retriever.retrieve(query)
                
                if not response:
                    logger.warning("Retriever returned empty response")
                    return []
                
                for doc in response:
                    contexts.append(doc.get_content())
                
                logger.info(f"Retrieved {len(contexts)} documents successfully")
                return contexts

            except Exception as e:
                logger.error(f"Retrieval attempt {attempt} failed: {e}")
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                
                if attempt < max_retries:
                    logger.info(f"Retrying in {backoff} seconds...")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    logger.error("All retrieval attempts failed, returning empty list")
                    return []
    
    def rerank(self, query: str, contexts: List[str]):
        """Reranks contexts based on relevance"""
        logger = logging.getLogger(__name__)
        try:
            logger.info(f"RERANK - Processing {len(contexts)} contexts")
            rerank_results = self.reranker.rerank(
                model='rerank-v3.5',
                query=query,
                documents=contexts
            )
            
            docs_reranked = [contexts[result.index] for result in rerank_results.results]
            logger.info(f"Reranking completed, returned {len(docs_reranked)} documents")
            return docs_reranked
        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            raise ValueError(f"Reranking failed: {e}")

    def combine_contexts(self, contexts: List[str]) -> str:
        """
        Combina i top contesti in un formato chiaro e strutturato
        """
        if not contexts:
            return "Nessun contesto disponibile."
        
        combined = "=== CONTESTI RILEVANTI ===\n\n"
        
        for i, context in enumerate(contexts, 1):
            combined += f"--- CONTESTO {i} ---\n"
            combined += f"{context.strip()}\n\n"
        
        combined += "=== FINE CONTESTI ==="
        
        return combined

    def query_expansion(self, query: str) -> List[str]:
        """Expands query using LLM"""
        logger = logging.getLogger(__name__)
        logger.info("QUERY EXPANSION - Generating sub-queries")
        time.sleep(1)
        
        try:
            system_prompt = '''Sei un assistente intelligente specializzato nell'analisi e nella riformulazione di query.  
            Data una singola "query originale", procedi nel modo seguente:

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
            
            response = self.llm.chat.completions.create(
                model=cfg.LLM_LLAMA_70B,
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
            logger.info(f"Query expansion generated {len(response)} sub-queries")
            return response
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}", exc_info=True)
            raise ValueError(f"Errore nella query expansion: {e}")
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generates predefined answer based on query and context"""
        logger = logging.getLogger(__name__)
        logger.info("GENERATE ANSWER - Creating response")
        time.sleep(1)
        
        # Preparazione del prompt
        system_prompt = """
        Analizza il contesto e scegli la risposta più appropriata:

        1. "Si" - L'organizzazione HA implementato completamente quanto richiesto con processi formali e documentati
        2. "No" - L'organizzazione NON ha implementato quanto richiesto o non ci sono evidenze
        3. "Si, ma senza una struttura ben definita" - L'organizzazione fa queste attività MA in modo informale, parziale, non documentato o non sistematico
        4. "Si, ma con controlli che avvengono in oltre un anno" - Le pratiche e i controlli richiesti sono svolti per un periodo superiore ad un anno
        
        CRITERI per "Si, ma senza una struttura ben definita":
        - ✓ L'attività viene svolta MA senza procedure scritte
        - ✓ Implementazione parziale o inconsistente
        - ✓ Manca documentazione formale
        - ✓ Non c'è un processo ripetibile
        - ✓ Dipende da singole persone, non da processi

        CRITERI per "Si, ma con controlli che avvengono in oltre un anno":
        - ✓ Le pratiche e i controlli richiesti sono svolti per un periodo superiore ad un anno
        - ✓ Si riferisce a domande di tipo temporale (es. "Fate dei review annuali di...")

        Esempi:
        - Query: "Fate security review?"
        Contesto: "Il team fa review quando si ricorda"
        Risposta: "Si, ma senza una struttura ben definita"
        
        - Query: "Avete un processo di incident response?"
        Contesto: "Gestiamo gli incidenti ma non abbiamo procedure scritte"
        Risposta: "Si, ma senza una struttura ben definita"
        
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
            response = self.llm.chat.completions.create(
                model=cfg.LLM_LLAMA_70B,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\n Context: {context}"}
                ],
                temperature=cfg.LLM_TEMPERATURE,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            if not isinstance(result, dict) or 'response' not in result or 'justification' not in result:
                raise ValueError("JSON di risposta non valido o incompleto")
            
            logger.info(f"Generated answer: {result['response']}")
            return result['response'], result['justification']
        except Exception as e:
            logger.error(f"Answer generation failed: {e}", exc_info=True)
            raise ValueError(f"Errore durante la generazione della risposta: {e}")

    def compile_and_save(self, question: str, answer: str, context: str, justification: str):
        """Saves Q&A pair to Excel file"""
        logger = logging.getLogger(__name__)
        logger.info(f"COMPILE AND SAVE - Saving question {self.question_id}")
        
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
            
            logger.info(f"Successfully saved Q&A to {excel_file}")
            
        except Exception as e:
            logger.error(f"Error saving to Excel: {e}", exc_info=True)
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
    
    def human_context_selection(self, question: str, contexts: List[str]) -> str:
        """
        Permette all'utente di selezionare il contesto più rilevante
        """
        logger = logging.getLogger(__name__)
        logger.info("HUMAN CONTEXT SELECTION - Waiting for user input")
        
        print("\n" + "="*80)
        print(f"DOMANDA: {question}")
        print("="*80)
        print("\nContesti disponibili:")
        print("-"*40)
        
        # Mostra i contesti con numerazione
        for i, context in enumerate(contexts, 1):
            print(f"\n[{i}] CONTESTO {i}:")
            print(f"{context[:500]}..." if len(context) > 500 else context)
            print("-"*40)
        
        # Opzioni aggiuntive
        print(f"\n[{len(contexts) + 1}] Combina TUTTI i contesti")
        print(f"[{len(contexts) + 2}] Salta questa domanda")
        print(f"[0] Mostra contesti completi")
        
        while True:
            try:
                choice = input(f"\nScegli il contesto più rilevante (1-{len(contexts) + 2}, 0 per dettagli): ").strip()
                
                if choice == "0":
                    # Mostra contesti completi
                    self._show_full_contexts(contexts)
                    continue
                    
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(contexts):
                    selected = contexts[choice_num - 1]
                    logger.info(f"User selected context {choice_num}")
                    return selected
                    
                elif choice_num == len(contexts) + 1:
                    # Combina tutti i contesti
                    combined = self.combine_contexts(contexts)
                    logger.info("User chose to combine all contexts")
                    return combined
                    
                elif choice_num == len(contexts) + 2:
                    # Salta la domanda
                    logger.info("User chose to skip question")
                    return "SKIPPED_BY_USER"
                    
                else:
                    print(f"Scelta non valida. Inserisci un numero tra 1 e {len(contexts) + 2}")
                    
            except ValueError:
                print("Inserisci un numero valido")
            except KeyboardInterrupt:
                print("\nOperazione interrotta dall'utente")
                logger.info("User interrupted context selection")
                return "INTERRUPTED_BY_USER"

    def _show_full_contexts(self, contexts: List[str]):
        """Mostra i contesti completi per una valutazione dettagliata"""
        print("\n" + "="*80)
        print("CONTESTI COMPLETI:")
        print("="*80)
        
        for i, context in enumerate(contexts, 1):
            print(f"\n[{i}] CONTESTO COMPLETO {i}:")
            print(context)
            print("-"*80)
            
        input("\nPremi ENTER per tornare alla selezione...")










