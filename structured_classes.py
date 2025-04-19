
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional

# Definizione dei modelli Pydantic per la struttura dell'output
class ExpandedQuery(BaseModel):
    original_query: str = Field(..., description="La query originale dell'utente")
    expanded_query: str = Field(..., description=" query espansa per migliorare il recupero")


# Definizione dell'Enum per i valori predefiniti
class PredefinedAnswer(str, Enum):
    POSITIVE = "Si"
    NEGATIVE = "No"
    SEMI_POSITIVE = "Si, ma senza una struttura ben definita"

# Modello Pydantic per la risposta
class ResponseGenerator(BaseModel):
    selected_response: PredefinedAnswer = Field(
        ..., 
        description="Seleziona una delle risposte predefinite che meglio si adatta alla query dell'utente"
    )
    justification: Optional[str] = Field(
        None, 
        description="Spiegazione opzionale del perché è stata selezionata questa risposta"
    )

class RelevanceResponse(BaseModel):
    is_relevant: bool = Field(
        ..., 
        description="Indica se il contesto è rilevante per la query (true) o non lo è (false)"
    )
