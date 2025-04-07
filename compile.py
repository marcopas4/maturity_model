from typing import List
import json
from agent import Agent

def compile(agent: Agent) -> None:
    """
    Compile answers for maturity model questions using the provided agent.
    
    Args:
        agent (Agent): Agent instance to process questions and generate answers
    """
    MAX_ATTEMPTS = 3
    
    def reset_state():
        """Reset all state variables for new question"""
        return {
            'grade': False,
            'expanded': False,
            'loop_idx': 0,
            'context': None,
            'retrieved_docs': None
        }

    for idx, question in enumerate(agent.questions):
        print(f"\nProcessing question {idx + 1}/{len(agent.questions)}")
        
        # Reset state
        state = reset_state()
        
        # Initial retrieval with auto-merging
        try:
            state['retrieved_docs'] = agent.retrieve(query=question, mode="auto-merging")
        except Exception as e:
            print(f"Initial retrieval failed: {e}")
            agent.compile_and_save("Error: Initial retrieval failed")
            continue

        # Main processing loop
        while not state['grade']:  # Rimosso il controllo su expanded
            # Try each retrieved document
            while state['loop_idx'] < min(len(state['retrieved_docs']), MAX_ATTEMPTS):
                try:
                    content = state['retrieved_docs'][state['loop_idx']].get_content()
                    state['grade'] = agent.grade_context(context=content)
                    
                    if state['grade']:
                        state['context'] = content
                        break
                    
                    state['loop_idx'] += 1
                    
                except IndexError:
                    print(f"No more documents to process")
                    break
                except Exception as e:
                    print(f"Error during retrieval attempt {state['loop_idx'] + 1}: {e}")
                    state['loop_idx'] += 1

            # If no relevant context found and not yet expanded, try query expansion
            if not state['grade'] and not state['expanded']:
                try:
                    print("Trying query expansion...")
                    query_dict = agent.query_expansion(question)
                    expanded_query = _extract_answer(query_dict)
                    state['retrieved_docs'] = agent.retrieve(query=expanded_query, mode="auto-merging")
                    state['expanded'] = True
                    state['loop_idx'] = 0
                    continue  # Torna all'inizio del while per processare i nuovi documenti
                except Exception as e:
                    print(f"Query expansion failed: {e}")
                    break
            
            # Se siamo qui, abbiamo finito i tentativi o fallito l'espansione
            break

        # Generate and save answer
        try:
            if state['grade']:
                agent.context = state['context']
                agent.generate_answer()
                answer = _extract_answer(agent.answer)
            else:
                answer = "Insufficient context to provide a detailed answer"
            
            agent.compile_and_save(answer)
            agent.question_id += 1
            
        except Exception as e:
            print(f"Error processing question {idx + 1}: {e}")
            agent.compile_and_save(f"Error: {str(e)}")

def _extract_answer(response) -> str:
    """Extract answer text from JSON response or handle errors."""
    try:
        answer_dict = json.loads(response.content)
        return answer_dict.get('response', 'No response found')
    except json.JSONDecodeError:
        return 'Error parsing JSON response'
    except AttributeError:
        return 'Invalid response format'



