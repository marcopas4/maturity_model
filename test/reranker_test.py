"""
Implementazione di un reranker basato su Sentence Transformers
per sistemi RAG (Retrieval-Augmented Generation).
"""

import numpy as np
from typing import List, Dict, Union, Optional, Any
from sentence_transformers import CrossEncoder
import time
import logging

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentenceTransformerReranker:
    """
    Reranker che utilizza i modelli cross-encoder di Sentence Transformers
    per riordinare i documenti recuperati in base alla loro rilevanza rispetto alla query.
    """
    
    def __init__(
        self, 
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", 
        max_length: int = 512,
        batch_size: int = 16,
    ):
        """
        Inizializza il reranker con un modello cross-encoder.
        
        Args:
            model_name: Nome del modello cross-encoder da utilizzare
                        (es. "cross-encoder/ms-marco-MiniLM-L-6-v2")
            max_length: Lunghezza massima della sequenza
            batch_size: Dimensione del batch per l'inferenza
        """
        logger.info(f"Inizializzazione del reranker con il modello {model_name}")
        self.model = CrossEncoder(model_name, max_length=max_length)
        self.batch_size = batch_size
        
    def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: Optional[int] = None,
        content_key: str = "content",
        score_key: str = "score",
        original_score_key: str = "original_score",
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Riordina i documenti in base alla loro rilevanza rispetto alla query.
        
        Args:
            query: La query dell'utente
            documents: Lista di documenti da riordinare
            top_k: Numero di documenti da restituire dopo il reranking
            content_key: Chiave per accedere al contenuto del documento
            score_key: Chiave per memorizzare il nuovo punteggio
            original_score_key: Chiave per memorizzare il punteggio originale
            return_scores: Se True, include i punteggi nel risultato
            
        Returns:
            Lista di documenti riordinati
        """
        if not documents:
            logger.warning("Nessun documento fornito per il reranking")
            return []
        
        start_time = time.time()
        
        # Estrai i testi dei documenti
        texts = [doc[content_key] for doc in documents]
        
        # Crea le coppie query-documento
        query_doc_pairs = [(query, text) for text in texts]
        
        # Calcola i punteggi di rilevanza utilizzando il cross-encoder
        logger.info(f"Esecuzione del reranking su {len(texts)} documenti")
        scores = self.model.predict(
            query_doc_pairs, 
            batch_size=self.batch_size, 
            show_progress_bar=len(query_doc_pairs) > 100
        )
        
        # Aggiorna i documenti con i nuovi punteggi
        ranked_docs = []
        for idx, (doc, score) in enumerate(zip(documents, scores)):
            # Crea una copia del documento per non modificare l'originale
            ranked_doc = doc.copy()
            
            # Salva il punteggio originale se esiste
            if score_key in ranked_doc:
                ranked_doc[original_score_key] = ranked_doc[score_key]
            
            # Aggiorna con il nuovo punteggio
            if return_scores:
                ranked_doc[score_key] = float(score)
            
            ranked_docs.append(ranked_doc)
        
        # Ordina i documenti in base al punteggio (in ordine decrescente)
        ranked_docs = sorted(ranked_docs, key=lambda x: x[score_key], reverse=True)
        
        # Limita il numero di risultati se richiesto
        if top_k is not None and top_k > 0:
            ranked_docs = ranked_docs[:top_k]
        
        elapsed_time = time.time() - start_time
        logger.info(f"Reranking completato in {elapsed_time:.2f} secondi")
        
        return ranked_docs


class RAGSystem:
    """
    Sistema RAG completo che integra un retriever, un reranker e un generatore.
    """
    
    def __init__(
        self,
        retriever,
        reranker = None,
        generator = None,
        top_k_retrieval: int = 20,
        top_k_rerank: int = 5,
        use_reranker: bool = True
    ):
        """
        Inizializza un sistema RAG completo.
        
        Args:
            retriever: Componente per il retrieval iniziale
            reranker: Componente per il reranking (opzionale)
            generator: Componente per la generazione di risposte
            top_k_retrieval: Numero di documenti da recuperare inizialmente
            top_k_rerank: Numero di documenti da mantenere dopo il reranking
            use_reranker: Se False, il reranker viene ignorato anche se disponibile
        """
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.use_reranker = use_reranker
        
    def answer(self, query: str) -> Dict[str, Any]:
        """
        Elabora una query attraverso il pipeline RAG.
        
        Args:
            query: La query dell'utente
            
        Returns:
            Dizionario con i risultati e metadati
        """
        # Retrieval iniziale
        logger.info(f"Esecuzione del retrieval per la query: {query}")
        start_retrieval = time.time()
        retrieved_docs = self.retriever.search(query, top_k=self.top_k_retrieval)
        retrieval_time = time.time() - start_retrieval
        
        # Reranking (se disponibile e abilitato)
        if self.reranker is not None and self.use_reranker:
            logger.info("Applicazione del reranking")
            start_rerank = time.time()
            reranked_docs = self.reranker.rerank(
                query=query, 
                documents=retrieved_docs, 
                top_k=self.top_k_rerank
            )
            rerank_time = time.time() - start_rerank
        else:
            logger.info("Reranking saltato")
            # Se il reranking è disabilitato, limita i documenti recuperati
            reranked_docs = retrieved_docs[:self.top_k_rerank]
            rerank_time = 0
        
        # Generazione della risposta (se il generatore è disponibile)
        if self.generator is not None:
            logger.info("Generazione della risposta")
            start_generation = time.time()
            # Estrai i contenuti per la generazione
            context_docs = reranked_docs if self.use_reranker and self.reranker else retrieved_docs[:self.top_k_rerank]
            context = "\n\n".join([doc["content"] for doc in context_docs])
            response = self.generator.generate(query=query, context=context)
            generation_time = time.time() - start_generation
        else:
            response = None
            generation_time = 0
        
        # Costruisci il risultato
        return {
            "query": query,
            "retrieved_documents": retrieved_docs,
            "reranked_documents": reranked_docs if self.use_reranker and self.reranker else None,
            "response": response,
            "metrics": {
                "retrieval_time": retrieval_time,
                "rerank_time": rerank_time,
                "generation_time": generation_time,
                "total_time": retrieval_time + rerank_time + generation_time
            }
        }


# Funzione di dimostrazione
def demonstrate_cross_encoder_reranker():
    """
    Funzione dimostrativa per l'utilizzo del reranker con Sentence Transformers.
    """
    # Simula un retriever semplice
    class DemoRetriever:
        def __init__(self):
            self.documents = [
                {
                    "id": "doc1",
                    "content": "I modelli cross-encoder sono progettati per valutare direttamente la rilevanza tra una query e un documento.",
                    "score": 0.75
                },
                {
                    "id": "doc2",
                    "content": "Sentence Transformers offre vari modelli cross-encoder pre-addestrati, come ms-marco-MiniLM-L-6-v2.",
                    "score": 0.70
                },
                {
                    "id": "doc3",
                    "content": "I reranker migliorano la qualità dei risultati di ricerca riordinando i documenti in base alla rilevanza semantica.",
                    "score": 0.65
                },
                {
                    "id": "doc4",
                    "content": "RAG (Retrieval-Augmented Generation) combina il retrieval di documenti con la generazione di testo per risposte più informative.",
                    "score": 0.60
                },
                {
                    "id": "doc5",
                    "content": "La differenza principale tra bi-encoder e cross-encoder è che i primi codificano query e documenti separatamente, mentre i secondi li valutano insieme.",
                    "score": 0.55
                },
                {
                    "id": "doc6",
                    "content": "La fotosintesi è il processo mediante il quale le piante convertono la luce solare in energia chimica.",
                    "score": 0.50
                },
                {
                    "id": "doc7",
                    "content": "Python è un linguaggio di programmazione ad alto livello, interpretato e con tipizzazione dinamica.",
                    "score": 0.45
                },
                {
                    "id": "doc8",
                    "content": "Le reti neurali trasformer hanno rivoluzionato l'elaborazione del linguaggio naturale grazie al meccanismo di attenzione.",
                    "score": 0.40
                },
                {
                    "id": "doc9",
                    "content": "Il retrieval semantico si basa sulla comprensione del significato piuttosto che sulla corrispondenza esatta delle parole.",
                    "score": 0.35
                },
                {
                    "id": "doc10",
                    "content": "I sistemi di raccomandazione suggeriscono contenuti agli utenti in base ai loro interessi e comportamenti passati.",
                    "score": 0.30
                }
            ]
        
        def search(self, query, top_k=5):
            # In una situazione reale, eseguiremmo una ricerca vettoriale
            # Per semplicità, restituiamo i documenti predefiniti
            return self.documents[:top_k]
    
    # Simula un generatore semplice
    class DemoGenerator:
        def generate(self, query, context):
            return f"Questa è una risposta generata per la query '{query}' utilizzando il contesto fornito dai documenti recuperati e riordinati."
    
    # Inizializza i componenti
    retriever = DemoRetriever()
    
    # Crea il reranker con Sentence Transformers
    reranker = SentenceTransformerReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=8
    )
    
    # Configurazione alternativa con un modello più grande per maggiore accuratezza
    # reranker = SentenceTransformerReranker(
    #     model_name="cross-encoder/ms-marco-TinyBERT-L-6-v1",
    #     batch_size=4
    # )
    
    # Inizializza il sistema RAG
    generator = DemoGenerator()
    rag_system = RAGSystem(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        top_k_retrieval=10,
        top_k_rerank=3
    )
    
    # Test del sistema
    query = "Come funzionano i cross-encoder nei sistemi di reranking?"
    print(f"Query: {query}\n")
    
    # Esegui il pipeline RAG completo
    result = rag_system.answer(query)
    
    # Mostra i documenti iniziali
    print("Documenti recuperati inizialmente:")
    for i, doc in enumerate(result["retrieved_documents"]):
        print(f"{i+1}. ID: {doc['id']} - Score: {doc['score']:.4f}")
        print(f"   {doc['content']}\n")
    
    # Mostra i documenti riordinati
    if result["reranked_documents"]:
        print("Documenti dopo il reranking:")
        for i, doc in enumerate(result["reranked_documents"]):
            print(f"{i+1}. ID: {doc['id']} - Score: {doc['score']:.4f}")
            print(f"   {doc['content']}\n")
    
    # Mostra la risposta generata
    print(f"Risposta: {result['response']}\n")
    
    # Mostra le metriche di prestazione
    print("Metriche di prestazione:")
    for metric, value in result["metrics"].items():
        print(f"- {metric}: {value:.4f} secondi")

# Esempio di utilizzo con diversi modelli disponibili
def available_cross_encoder_models():
    """
    Elenca i principali modelli cross-encoder disponibili in Sentence Transformers.
    """
    return {
        "Modelli per ricerca generale": [
            "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Buon compromesso velocità/accuratezza
            "cross-encoder/ms-marco-MiniLM-L-12-v2", # Più accurato del precedente
            "cross-encoder/ms-marco-TinyBERT-L-6-v1", # Veloce ma meno accurato
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", # Multilingua
        ],
        "Modelli per casi specifici": [
            "cross-encoder/stsb-roberta-large",  # Ottimizzato per similarità semantica
            "cross-encoder/quora-roberta-large",  # Ottimizzato per il rilevamento di duplicati
            "cross-encoder/nli-deberta-v3-large", # Ottimizzato per inferenza naturale del linguaggio
            "cross-encoder/nli-deberta-v3-xsmall", # Versione più leggera per NLI
        ]
    }

# Esecuzione di esempio

    # Dimostra l'utilizzo del reranker
demonstrate_cross_encoder_reranker()
    
    # Mostra i modelli disponibili
models = available_cross_encoder_models()
print("\nPrincipali modelli cross-encoder disponibili:")
for category, model_list in models.items():
        print(f"\n{category}:")
        for model in model_list:
            print(f"- {model}")