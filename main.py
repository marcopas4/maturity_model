# main.py
import os
import agent
import config
import compile as cp
import utils.document_loader as doc_loader

def main():
    """Funzione principale per eseguire il grafo LangGraph."""
    # Assicurati che le directory necessarie esistano
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    abot = agent.Agent()
    
    # Costruisci ed esegui il grafo
    cp.compile(abot)
    
    
    print(f"Elaborazione completata. Risultati salvati in {config.RESPONSES_PATH}")
    
if __name__ == "__main__":
    main()