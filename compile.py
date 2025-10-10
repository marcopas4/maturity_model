from typing import List
import json
import logging
import os
from logging.handlers import RotatingFileHandler
import config as cfg
from agent import Agent

def setup_logging():
    """Configura logging con file rotante e console"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Evita duplicati se già configurato
    if logger.handlers:
        return
    
    # Crea directory per log se non esiste
    log_dir = os.path.join(cfg.RESULTS_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # File handler rotante (max 10MB, 5 backup files)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'maturity_model.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter dettagliato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def compile(agent: Agent) -> None:
    """
    Compile answers for maturity model questions using the provided agent.
    """
    setup_logging()  # Configura logging una sola volta
    logger = logging.getLogger(__name__)
    
    def reset_state():
        """Reset all state variables for new question"""
        return {
            'context': None,
            'retrieved_docs': [],  # FIX: inizializza come lista vuota
            'answer': None,
            'justification': None
        }

    for idx in range(122,len(agent.questions)):
        question = agent.questions[idx]
        logger.info(f"Processing question {idx + 1}/{len(agent.questions)}: {question[:100]}...")
        
        # Reset state for new question
        state = reset_state()
        
        # Initial retrieval
        try:
            logger.info("Starting initial retrieval")
            state['retrieved_docs'] = agent.retrieve(query=question)
            logger.info(f"Initial retrieval returned {len(state['retrieved_docs'])} documents")
            
            expanded_query = agent.query_expansion(question)
            logger.info(f"Query expansion generated {len(expanded_query)} sub-queries")

        except Exception as e:
            logger.error(f"Initial retrieval failed for question {idx + 1}: {e}", exc_info=True)
            agent.compile_and_save(question=question, answer="Error: Initial retrieval failed", context=None, justification=str(e))
            agent.question_id += 1
            continue
        
        # Additional retrievals for expanded queries
        try:
            for i, query in enumerate(expanded_query):
                logger.info(f"Processing expanded query {i+1}/{len(expanded_query)}")
                contexts = agent.retrieve(query=query)
                state['retrieved_docs'].extend(contexts)  # FIX: usa extend invece di +=
                logger.info(f"Added {len(contexts)} contexts from expanded query {i+1}")
        except Exception as e:
            logger.error(f"Expanded query retrieval failed: {e}", exc_info=True)
            # Continua con i documenti già recuperati
        
        try:
            initial_count = len(state['retrieved_docs'])
            state['retrieved_docs'] = agent.deduplicate_post_reranking(state['retrieved_docs'])
            logger.info(f"Deduplication: {initial_count} -> {len(state['retrieved_docs'])} documents")
            
            # Rerank documents
            state['retrieved_docs'] = agent.rerank(query=question, contexts=state['retrieved_docs'])
            if not state['retrieved_docs']:
                raise ValueError("No relevant documents found after reranking")
            logger.info(f"Reranked {len(state['retrieved_docs'])} documents for question {idx + 1}")
        except Exception as e:
            logger.error(f"Reranking failed for question {idx + 1}: {e}", exc_info=True)
            agent.compile_and_save(question, "Error: Reranking failed", None, str(e))
            agent.question_id += 1
            continue
            
        # Usa i top 5 contesti dal reranker concatenati
        try:
            # Prendi i top 5 contesti (o meno se non disponibili)
            top_contexts = state['retrieved_docs'][:5]
            logger.info(f"Using top {len(top_contexts)} contexts for question {idx + 1}")
            
            # Concatena i contesti in modo chiaro
            combined_context = agent.combine_contexts(top_contexts)
            
            # Genera risposta usando il contesto combinato
            state['answer'], state['justification'] = agent.generate_answer(query=question, context=combined_context)
            
            # Salva il risultato
            agent.compile_and_save(
                question=question, 
                answer=state['answer'], 
                justification=state['justification'], 
                context=combined_context
            )
            
            logger.info(f"Successfully answered question {idx + 1}")
            agent.question_id += 1
            
        except Exception as e:
            logger.error(f"Failed to process question {idx + 1}: {e}", exc_info=True)
            try:
                agent.compile_and_save(
                    question=question, 
                    answer="Error", 
                    justification=f"Processing failed: {str(e)}", 
                    context=None
                )
                agent.question_id += 1
            except Exception as save_error:
                logger.error(f"Failed to save error for question {idx + 1}: {save_error}", exc_info=True)







