import os,config as cfg,time
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font


def compile_and_save(question_id: int, question: str, answer: str):
    """
    Saves Q&A pair to an Excel file.
    
    Args:
        question_id (int): Question identifier
        question (str): Question text
        answer (str): Generated answer
            
    Creates Excel file with columns:
        - Question ID
        - Question
        - Answer
        - Timestamp
        
    File location: cfg.RESULTS_DIR/responses.xlsx
    """
    print("---COMPILE AND SAVE---")
    
    excel_file = os.path.join(cfg.RESULTS_DIR, 'responses.xlsx')
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    
    new_data = pd.DataFrame({
        'Question ID': [question_id],
        'Question': [question],
        'Answer': [answer],
        'Timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')]
    })
    
    try:
        if os.path.isfile(excel_file):
            existing_df = pd.read_excel(excel_file)
            updated_df = pd.concat([existing_df, new_data], ignore_index=True)
        else:
            updated_df = new_data
            
        # Create a new Excel writer object
        writer = pd.ExcelWriter(excel_file, engine='openpyxl')
        
        # Write the DataFrame without index
        updated_df.to_excel(writer, sheet_name='Responses', index=False)
        
        # Access the worksheet
        workbook = writer.book
        worksheet = workbook['Responses']
        
        # Set column widths
        column_widths = {
            'A': 12,  # Question ID
            'B': 50,  # Question
            'C': 20,  # Answer
            'D': 20   # Timestamp
        }
        
        for col, width in column_widths.items():
            worksheet.column_dimensions[col].width = width
        
        # Style header row
        header_font = Font(bold=True)
        for cell in worksheet[1]:
            cell.font = header_font
            
        # Save and close
        writer.close()
        
        print(f"Saved Q&A to {excel_file}")
        
    except Exception as e:
        print(f"Error while handling Excel file: {e}")
        raise

question_id = 1
question = "What is the capital of France?"
answer = "The capital of France is Paris."
compile_and_save(question_id, question, answer)