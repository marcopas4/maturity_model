import os
from typing import Optional
from pydantic import BaseModel, Field
from openai import OpenAI
import config as cfg
# Modello Pydantic per la risposta di rilevanza
class RelevanceResponse(BaseModel):
    is_relevant: bool = Field(
        ..., 
        description="Indica se il contesto è rilevante per la query (true) o non lo è (false)"
    )

class ContextRelevanceChecker:
    def __init__(self, api_key=None, base_url=None):
        """
        Inizializza il valutatore di rilevanza.
        
        Args:
            api_key: API key per Groq o OpenAI
            base_url: URL base per l'API (es. "https://api.groq.com/openai/v1" per Groq)
        """
        self.client = OpenAI(
            api_key=api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or ("https://api.groq.com/openai/v1" if os.environ.get("GROQ_API_KEY") else None)
        )
    
    def check_relevance(self, query: str, context: str, model: str = cfg.LLM_MODEL) -> bool:
        """
        Verifica se il contesto è rilevante per la query.
        
        Args:
            query: La query dell'utente
            context: Il contesto da valutare
            model: Il modello da utilizzare
            
        Returns:
            bool: True se il contesto è rilevante, False altrimenti
        """
        # Definizione dello schema della funzione
        function_schema = {
            "name": "evaluate_context_relevance",
            "description": "Valuta se il contesto fornito è rilevante per rispondere alla query",
            "parameters": RelevanceResponse.model_json_schema()
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
            response = self.client.chat.completions.create(
                model=model,
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
            result = RelevanceResponse.model_validate_json(function_args)
            
            return result.is_relevant
        
        except Exception as e:
            raise RuntimeError(f"Errore durante la valutazione della rilevanza: {str(e)}")

# Esempio di utilizzo

    # Inizializza il checker
checker = ContextRelevanceChecker()
    
    # Esempi di query e contesti
examples = [
        {
            "query": "Quali sono i principali effetti del riscaldamento globale?",
            "context": "Il riscaldamento globale sta causando l'innalzamento del livello del mare, eventi meteorologici estremi più frequenti e cambiamenti negli ecosistemi. Le temperature medie globali sono aumentate di circa 1°C dal periodo preindustriale."
        },
        {
            "query": "Quali sono i principali effetti del riscaldamento globale?",
            "context": "La pizza napoletana si prepara con farina di tipo 00, acqua, sale e lievito. Deve essere cotta in forno a legna a temperature molto elevate per circa 90 secondi."
        },
        {
            "query": "Come funzionano i token in NLP?",
            "context": "In Natural Language Processing, i token sono unità di testo come parole o sottoinsiemi di parole. La tokenizzazione è il processo di divisione del testo in queste unità per l'elaborazione da parte dei modelli linguistici."
        }
    ]
    
    # Verifica la rilevanza per ogni esempio
for i, example in enumerate(examples):
        is_relevant = checker.check_relevance(example["query"], example["context"])
        print(f"Esempio {i+1}:")
        print(f"Query: {example['query']}")
        print(f"Contesto: {example['context'][:100]}..." if len(example['context']) > 100 else f"Contesto: {example['context']}")
        print(f"Rilevante: {is_relevant}")
        print("-" * 50)