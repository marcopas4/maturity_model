# main.py
import os
import agent
import config
import compile as cp
import utils.document_loader as doc_loader
from gui import launch_gui

def main():
    """Funzione principale per eseguire il grafo LangGraph."""
    # Assicurati che le directory necessarie esistano
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    abot = agent.Agent()
    print(f"Reranker: {type(abot.reranker).__name__}")
    
    # Scegli modalità: GUI o console
    mode = input("Scegli modalità: [1] GUI (Gradio) [2] Console: ").strip()
    
    if mode == "1":
        print("🚀 Avviando interfaccia Gradio...")
        print("L'interfaccia sarà disponibile su: http://127.0.0.1:7860")
        launch_gui(abot)
    else:
        # Modalità console originale
        cp.compile(abot)
        print(f"Elaborazione completata. Risultati salvati in {config.RESPONSES_PATH}")
    
if __name__ == "__main__":
    main()