from typing import List
import json
import logging
from agent import Agent

def compile(agent: Agent) -> None:
    """
    Compile answers for maturity model questions using the provided agent.
    """
    MAX_ATTEMPTS = 5
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def reset_state():
        """Reset all state variables for new question"""
        return {
            'grade': False,
            'attempt_count': 0,
            'context': None,
            'retrieved_docs': None,
            'answer': None,
            'justification':None
        }

    for idx, question in enumerate(agent.questions):
        logger.info(f"Processing question {idx + 1}/{len(agent.questions)}")
        
        # Reset state for new question
        state = reset_state()
        
        # Initial retrieval
        try:
            state['retrieved_docs'] = agent.retrieve(query=question)
            expanded_query = agent.query_expansion(question)

        except Exception as e:
            logger.error(f"Initial retrieval failed: {e}")
            agent.compile_and_save(question=question,answer="Error: Initial retrieval failed",context= None, justification=None)
            agent.question_id += 1
            continue
        
        for query in expanded_query:
            contexts = agent.retrieve(query=query)
            state['retrieved_docs'] += contexts
        
        try:
            state['retrieved_docs'] = agent.deduplicate_post_reranking(state['retrieved_docs'])  # Remove duplicates
            #rerank documents
            state['retrieved_docs'] = agent.rerank(query=question,contexts=state['retrieved_docs'])
            if not state['retrieved_docs']:
                raise ValueError("No relevant documents found after reranking")
            logger.info(f"Reranked {len(state['retrieved_docs'])} documents for question {idx + 1}")
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            agent.compile_and_save(question,"Error: Reranking failed", None,None)
            agent.question_id += 1
            continue
        #grade question
        while not state['grade'] and state['attempt_count'] < MAX_ATTEMPTS:
            try:
                state['grade'] = agent.grade_context(query=question,context=state['retrieved_docs'][state['attempt_count']])
                if state['grade']:
                    state['context'] = state['retrieved_docs'][state['attempt_count']]
                    state['answer'],state['justification'] = agent.generate_answer(query=question,context=state['context'])
                    agent.compile_and_save(question=question,answer=state['answer'], justification=state['justification'],context=state['context'])
                    logger.info(f"Successfully answered question {idx + 1}")
                    agent.question_id += 1
                    break
                else:
                    state['attempt_count'] += 1
            except Exception as e:
                logger.error(f"Attempt {state['attempt_count'] + 1} failed: {e}")
                state['attempt_count'] += 1
                
        if not state['grade']:
            try:
                agent.compile_and_save(question=question,answer="No", justification="No relevant documents found", context=None)
                agent.question_id += 1
            except Exception as e:
                logger.error(f"Failed to save answer for question {idx + 1}: {e}")







