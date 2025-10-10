import gradio as gr
import logging
from typing import List, Tuple, Optional
from agent import Agent
import config as cfg

class MaturityModelGUI:
    """Interfaccia grafica per il modello di maturità usando Gradio"""
    
    def __init__(self, agent: Agent):
        self.agent = agent
        self.logger = logging.getLogger(__name__)
        self.current_question_idx = 0
        self.current_contexts = []
        self.current_question = ""
        self.selected_context = None
        self.question_answered = False  # ← NUOVO: traccia se la domanda corrente è stata già risposta
        
    def get_current_question(self) -> Tuple[str, str]:
        """Restituisce la domanda corrente e il progresso"""
        if self.current_question_idx < len(self.agent.questions):
            self.current_question = self.agent.questions[self.current_question_idx]
            progress = f"Domanda {self.current_question_idx + 1} di {len(self.agent.questions)}"
            return self.current_question, progress
        else:
            return "Tutte le domande completate!", "Completato"
    
    def process_question(self) -> Tuple[List[str], str]:
        """Processa la domanda corrente e restituisce i contesti"""
        if self.current_question_idx >= len(self.agent.questions):
            return [], "Processo completato"
        
        try:
            question = self.current_question
            self.logger.info(f"Processing question {self.current_question_idx + 1}: {question}")
            
            # Recupera documenti
            retrieved_docs = self.agent.retrieve(query=question)
            
            # Espansione query
            expanded_queries = self.agent.query_expansion(question)
            for expanded_query in expanded_queries:
                contexts = self.agent.retrieve(query=expanded_query)
                retrieved_docs.extend(contexts)
            
            # Deduplicazione e reranking
            retrieved_docs = self.agent.deduplicate_post_reranking(retrieved_docs)
            retrieved_docs = self.agent.rerank(query=question, contexts=retrieved_docs)
            
            # Prendi i top 5
            self.current_contexts = retrieved_docs[:5]
            
            status = f"Trovati {len(self.current_contexts)} contesti rilevanti"
            return self.current_contexts, status
            
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            return [], f"Errore: {str(e)}"
    
    def select_context(self, context_choice: str) -> Tuple[str, str, str]:
        """Seleziona un contesto e genera la risposta"""
        if not self.current_contexts:
            return "", "", "Nessun contesto disponibile"
        
        try:
            if context_choice == "Combina tutti":
                selected_context = self.agent.combine_contexts(self.current_contexts)
            elif context_choice == "Salta domanda":
                # Salva come saltata (o sovrascrive se già risposta)
                self._save_or_update_answer("SKIPPED", "Domanda saltata dall'utente", "N/A")
                return "SKIPPED", "Domanda saltata dall'utente", "Domanda saltata (salvata)"
            else:
                # Estrai il numero del contesto
                try:
                    context_num = int(context_choice.split()[1]) - 1  # "Contesto 1" -> index 0
                    if 0 <= context_num < len(self.current_contexts):
                        selected_context = self.current_contexts[context_num]
                    else:
                        return "", "", "Selezione contesto non valida"
                except (ValueError, IndexError):
                    return "", "", "Formato selezione non valido"
            
            # Genera risposta
            answer, justification = self.agent.generate_answer(
                query=self.current_question,
                context=selected_context
            )
            
            # Salva o aggiorna risultato
            self._save_or_update_answer(answer, justification, selected_context)
            
            status_msg = "Risposta aggiornata" if self.question_answered else "Risposta generata con successo"
            return answer, justification, status_msg
            
        except Exception as e:
            self.logger.error(f"Error selecting context: {e}")
            return "", "", f"Errore: {str(e)}"
    
    def _save_or_update_answer(self, answer: str, justification: str, context: str):
        """Salva una nuova risposta o aggiorna quella esistente"""
        if not self.question_answered:
            # Prima volta che rispondiamo a questa domanda
            self.agent.compile_and_save(
                question=self.current_question,
                answer=answer,
                justification=justification,
                context=context
            )
            self.agent.question_id += 1
            self.question_answered = True
            self.logger.info(f"New answer saved for question {self.current_question_idx + 1}")
        else:
            # Domanda già risposta, aggiorna la riga esistente
            self._update_existing_answer(answer, justification, context)
            self.logger.info(f"Answer updated for question {self.current_question_idx + 1}")
    
    def _update_existing_answer(self, answer: str, justification: str, context: str):
        """Aggiorna la risposta esistente nel file Excel"""
        import pandas as pd
        import os
        import time
        
        excel_file = os.path.join(cfg.RESULTS_DIR, 'responses.xlsx')
        
        try:
            if os.path.isfile(excel_file):
                # Leggi il file esistente
                df = pd.read_excel(excel_file)
                
                # Trova la riga corrispondente alla domanda corrente (question_id - 1 perché è già stato incrementato)
                current_question_id = self.agent.question_id - 1
                mask = df['Question ID'] == current_question_id
                
                if mask.any():
                    # Aggiorna la riga esistente
                    df.loc[mask, 'Answer'] = answer
                    df.loc[mask, 'Justification'] = justification
                    df.loc[mask, 'Context'] = context
                    df.loc[mask, 'Timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Salva il file aggiornato
                    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='Responses', index=False)
                        
                        # Applica gli stili
                        workbook = writer.book
                        worksheet = workbook['Responses']
                        
                        # Set column widths
                        column_widths = {
                            'A': 12, 'B': 50, 'C': 20, 'D': 50, 'E': 40, 'F': 20
                        }
                        for col, width in column_widths.items():
                            worksheet.column_dimensions[col].width = width
                        
                        # Style header row
                        from openpyxl.styles import Font
                        header_font = Font(bold=True)
                        for cell in worksheet[1]:
                            cell.font = header_font
                    
                    self.logger.info(f"Successfully updated answer in {excel_file}")
                else:
                    self.logger.warning(f"Question ID {current_question_id} not found for update")
            else:
                self.logger.warning("Excel file not found for update")
                
        except Exception as e:
            self.logger.error(f"Error updating Excel file: {e}")
    
    def next_question(self) -> Tuple[str, str, str, str, str, str]:
        """Passa alla domanda successiva e resetta l'interfaccia"""
        self.current_question_idx += 1
        self.current_contexts = []
        self.selected_context = None
        self.question_answered = False  # ← RESET: la nuova domanda non è ancora stata risposta
        
        # Ottieni nuova domanda
        question, progress = self.get_current_question()
        
        # Reset tutti i campi
        return (
            question,           # question_display
            progress,           # progress_display  
            "",                 # context_choice (reset dropdown)
            "",                 # answer_display
            "",                 # justification_display
            "Pronto per la prossima domanda"  # status_display
        )
    
    def create_interface(self):
        """Crea l'interfaccia Gradio"""
        with gr.Blocks(title="Maturity Model Assessment", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🎯 Maturity Model Assessment Tool")
            gr.Markdown("Valuta la maturità della tua organizzazione rispondendo alle domande con l'aiuto dell'AI")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Sezione Domanda
                    gr.Markdown("## 📋 Domanda Corrente")
                    progress_display = gr.Textbox(
                        label="Progresso", 
                        value="Caricamento...", 
                        interactive=False
                    )
                    question_display = gr.Textbox(
                        label="Domanda",
                        value="Caricamento...",
                        lines=3,
                        interactive=False
                    )
                    
                    # Pulsante per processare la domanda
                    process_btn = gr.Button("🔄 Carica Contesti", variant="primary")
                    
                with gr.Column(scale=1):
                    # Controlli
                    gr.Markdown("## ⚙️ Controlli")
                    status_display = gr.Textbox(
                        label="Status",
                        value="Pronto",
                        interactive=False
                    )
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Sezione Contesti
                    gr.Markdown("## 📚 Contesti Rilevanti")
                    with gr.Accordion("Contesti Trovati", open=True):
                        context_1 = gr.Textbox(label="Contesto 1", lines=4, interactive=False)
                        context_2 = gr.Textbox(label="Contesto 2", lines=4, interactive=False)
                        context_3 = gr.Textbox(label="Contesto 3", lines=4, interactive=False)
                        context_4 = gr.Textbox(label="Contesto 4", lines=4, interactive=False)
                        context_5 = gr.Textbox(label="Contesto 5", lines=4, interactive=False)
                
                with gr.Column(scale=1):
                    # Selezione contesto
                    gr.Markdown("## 🎯 Selezione")
                    context_choice = gr.Dropdown(
                        label="Scegli il contesto",
                        choices=[],
                        interactive=True
                    )
                    select_btn = gr.Button("✅ Seleziona e Genera Risposta", variant="secondary")
                    gr.Markdown("*💡 Puoi cliccare più volte per aggiornare la risposta*")
            
            with gr.Row():
                with gr.Column():
                    # Sezione Risposta
                    gr.Markdown("## 🤖 Risposta AI")
                    answer_display = gr.Textbox(
                        label="Risposta",
                        lines=2,
                        interactive=False
                    )
                    justification_display = gr.Textbox(
                        label="Giustificazione",
                        lines=4,
                        interactive=False
                    )
                    
                    next_btn = gr.Button("➡️ Domanda Successiva", variant="primary", size="lg")
            
            # Stato interno (nascosto)
            contexts_state = gr.State([])
            
            # Callback functions (resta uguale al codice precedente)
            def process_question_callback():
                """Callback per processare la domanda"""
                contexts, status = self.process_question()
                
                # Prepara la visualizzazione dei contesti
                displays = ["", "", "", "", ""]
                choices = []
                
                for i, context in enumerate(contexts[:5]):
                    displays[i] = context
                    choices.append(f"Contesto {i+1}")
                
                # Aggiungi opzioni speciali
                if contexts:
                    choices.extend(["Combina tutti", "Salta domanda"])
                
                return (
                    displays[0],    # context_1
                    displays[1],    # context_2
                    displays[2],    # context_3
                    displays[3],    # context_4
                    displays[4],    # context_5
                    gr.Dropdown(choices=choices, value=""),  # context_choice aggiornato
                    status,         # status_display
                    contexts        # contexts_state
                )
            
            def select_context_callback(choice, contexts):
                """Callback per selezionare il contesto"""
                self.current_contexts = contexts  # Ripristina lo stato
                return self.select_context(choice)
            
            def next_question_callback():
                """Callback per la domanda successiva"""
                result = self.next_question()
                
                # Reset dei contesti
                empty_contexts = ["", "", "", "", ""]
                empty_dropdown = gr.Dropdown(choices=[], value="")
                
                return result + tuple(empty_contexts) + (empty_dropdown,)
            
            # Event handlers (resto uguale)
            process_btn.click(
                fn=process_question_callback,
                outputs=[context_1, context_2, context_3, context_4, context_5, 
                        context_choice, status_display, contexts_state]
            )
            
            select_btn.click(
                fn=select_context_callback,
                inputs=[context_choice, contexts_state],
                outputs=[answer_display, justification_display, status_display]
            )
            
            next_btn.click(
                fn=next_question_callback,
                outputs=[question_display, progress_display, context_choice,
                        answer_display, justification_display, status_display,
                        context_1, context_2, context_3, context_4, context_5, context_choice]
            )
            
            # Inizializza la prima domanda al caricamento
            interface.load(
                fn=self.get_current_question,
                outputs=[question_display, progress_display]
            )
        
        return interface

def launch_gui(agent: Agent):
    """Lancia l'interfaccia grafica"""
    gui = MaturityModelGUI(agent)
    interface = gui.create_interface()
    
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True
    )