import pandas as pd
import logging
from typing import Dict, List, Optional
import config

class QuestionGen:
    """
    Classe per generare domande da un file Excel di modello di maturità.
    Gestisce il caricamento, la pulizia e l'estrazione delle domande dal file.
    """
    
    def __init__(self, excel_path: str = config.QUESTIONS_PATH):
        """
        Inizializza l'oggetto QuestionGen.
        
        Args:
            excel_path: Percorso del file Excel contenente le domande.
        """
        self.excel_path = excel_path
        self.sheet_names = [
            'Governance', 
            'Architecture & Design', 
            'Code Development & Review', 
            'Build & Deployment', 
            'Test & Verification', 
            'Operations & Observability'
        ]
        self.logger = logging.getLogger(__name__)
        
    def load_file(self) -> Dict[str, pd.DataFrame]:
        """
        Carica il file Excel e pulisce i dati.
        
        Returns:
            Un dizionario contenente i DataFrame puliti per ogni foglio.
        
        Raises:
            FileNotFoundError: Se il file Excel non esiste.
            KeyError: Se un foglio richiesto non esiste nel file.
        """
        try:
            # Carica solo i fogli specificati
            dfs = pd.read_excel(
                self.excel_path, 
                sheet_name=self.sheet_names
            )
            
            # Pulisci i dati per ogni foglio
            for sheet_name, df in dfs.items():
                # Elimina colonne in base al numero di colonne disponibili
                num_cols = len(df.columns)
                if num_cols >= 10:
                    df = df.drop(df.columns[[4, 5, 6, 7, 8, 9]], axis=1, errors='ignore')
                elif num_cols >= 9:
                    df = df.drop(df.columns[[4, 5, 6, 7, 8]], axis=1, errors='ignore')
                else:
                    df = df.drop(df.columns[[4, 5, 6, 7]], axis=1, errors='ignore')
                
                # Rinomina le colonne usando i valori della prima riga
                if not df.empty:
                    new_cols = {df.columns[i]: str(df.iloc[0, i]) for i in range(len(df.columns))}
                    df = df.rename(columns=new_cols)
                    
                    # Rimuovi la prima riga e reimposta l'indice
                    df = df.drop(index=0).reset_index(drop=True)
                
                # Salva il DataFrame aggiornato
                dfs[sheet_name] = df
            
            return dfs
            
        except FileNotFoundError:
            self.logger.error(f"File Excel non trovato: {self.excel_path}")
            raise
        except KeyError as e:
            self.logger.error(f"Foglio non trovato nel file Excel: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Errore durante il caricamento del file: {e}")
            raise
    
    def display_dataframes(self, dfs: Dict[str, pd.DataFrame]) -> None:
        """
        Visualizza i primi 5 record di ogni DataFrame.
        
        Args:
            dfs: Dizionario contenente i DataFrame da visualizzare.
        """
        for sheet_name, df in dfs.items():
            self.logger.info(f"Foglio: {sheet_name}")
            self.logger.info(f"\n{df.head()}")
    
    def extract_all_questions(self) -> List[str]:
        """
        Estrae tutte le domande dal file Excel.
        
        Returns:
            Lista di stringhe, ciascuna contenente una domanda numerata.
        """
        # Carica i dati
        dataframes = self.load_file()
        
        # Estrai tutte le domande
        all_questions = []        
        for sheet_name, df in dataframes.items():
            if 'Evaluation Questions' not in df.columns:
                self.logger.warning(f"Colonna 'Evaluation Questions' non trovata nel foglio {sheet_name}")
                continue
                
            for _, row in df.iterrows():
                question = row.get('Evaluation Questions')
                if pd.notnull(question):  # Controlla se la domanda non è nulla
                    all_questions.append(f"{question}")
                    
        
        return all_questions


# Esempio di utilizzo
if __name__ == "__main__":
    # Configura il logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Crea un'istanza e genera le domande
    question_gen = QuestionGen()
    all_questions = question_gen.display_dataframes(question_gen.load_file())
    
    