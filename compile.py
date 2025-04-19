from typing import List
import json
import logging
from agent import Agent

def compile(agent: Agent) -> None:
    """
    Compile answers for maturity model questions using the provided agent.
    
    Args:
        agent (Agent): Agent instance to process questions and generate answers
    """
    MAX_ATTEMPTS = 5
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def reset_state():
        """Reset all state variables for new question"""
        return {
            'grade': False,
            'expanded': False,
            'attempt_count': 0,
            'context': None,
            'retrieved_docs': None,
            'answer': None 
        }

    for idx, question in enumerate(agent.questions):
        logger.info(f"Processing question {idx + 1}/{len(agent.questions)}")
        
        # Reset state for new question
        state = reset_state()
        
        # Initial retrieval with auto-merging
        try:
            state['retrieved_docs'] = agent.retrieve(query=question, mode="auto-merging")
        except Exception as e:
            logger.error(f"Initial retrieval failed: {e}")
            agent.compile_and_save("Error: Initial retrieval failed")
            agent.question_id += 1
            continue

        # Main processing loop - with total attempt limit
        while not state['grade'] and state['attempt_count'] < MAX_ATTEMPTS:
            doc_idx = 0
            
            # Try each retrieved document
            while doc_idx < len(state['retrieved_docs']) and not state['grade']:
                try:
                    content = state['retrieved_docs'][doc_idx].get_content()
                    state['grade'] = agent.grade_context(query=question, context=content)
                    
                    if state['grade']:
                        state['context'] = content
                        break
                    
                except Exception as e:
                    logger.error(f"Error processing document {doc_idx + 1}: {e}")
                
                doc_idx += 1
                state['attempt_count'] += 1
                
                # Exit if we've reached maximum attempts
                if state['attempt_count'] >= MAX_ATTEMPTS:
                    break

            # If no relevant context found and not yet expanded, try query expansion
            if not state['grade'] and not state['expanded'] :
                try:
                    logger.info("Trying query expansion...")
                     
                    expanded_query = agent.query_expansion(question)
                    state['retrieved_docs'] = agent.retrieve(query=expanded_query, mode="auto-merging")
                    state['expanded'] = True
                    # Count this as an attempt
                    state['attempt_count'] = 0
                    # Continue to process the new documents
                    continue
                except Exception as e:
                    logger.error(f"Query expansion failed: {e}")
                    break
            
            # If we reach here, either we've found a good context or exhausted all options
            break

        # Generate and save answer
        try:
            if state['grade'] and state['context']:
                answer = agent.generate_answer(context=state['context'], query=question)
                state['answer'] = answer
            else:
                answer = "Insufficient context to provide a detailed answer"
            
            agent.compile_and_save(state['answer'])
            
        except Exception as e:
            logger.error(f"Error processing question {idx + 1}: {e}")
            agent.compile_and_save(f"Error: {str(e)}")
        
        # Always increment question_id at the end
        agent.question_id += 1