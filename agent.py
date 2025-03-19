
from typing import List
from langchain.retrievers import ParentDocumentRetriever
from utils.questionGen import QuestionGen
import config as cfg
import utils.document_loader as dl
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import time
import pandas as pd
from langchain_core.runnables import RunnablePassthrough

class Agent:
    """
    Agent is the object that contains the state of the conversation
    """
    def __init__(self):
        self.curr_question: str = None  # qestionnaire question
        self.question_id: int = 0# question ID

        
        self.answer: str = None # Answer generated
        self.context: str =None  #  retrieved documents
        self.retriever :ParentDocumentRetriever = None # Retriever object
        self.conversation_history: List[dict] = []    # Storico della conversazione
        self.questions: List[str] = []
        self.llm = ChatGroq(
            model="deepseek-r1-distill-llama-70b",  
            temperature=0,  
            max_tokens=6000,
            timeout=None,
            max_retries=2,
            groq_api_key=cfg.GROQ_API_KEY
            )

    
        """
        Nodo di inizializzazione:
        1. Carica il questionario da un dataframe
        2. Inizializza i documenti per il retrieval RAG
        """
        try:
            print("---INIT---")
            # Estrae le domande dal file Excel
            
            self.questions = QuestionGen().extract_all_questions()
            
            # Inizializza il retirever and vectordb
            documents = dl.load_documents()

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                

                # Configura l'embedding model
            embed_model = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",  #sentence-transformers/all-mpnet-base-v2  BAAI/bge-small-en-v1.5
                    model_kwargs={'device': device}
                )
            
            index = faiss.IndexFlatL2(len(embed_model.embed_query("hello world")))
            vector_store = FAISS(
                    embedding_function=embed_model.embed_query,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                )
            child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=cfg.CHUNK_SIZE_CHILD,
                chunk_overlap=cfg.CHUNK_OVERLAP,
                length_function=len,
                )
            parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=cfg.CHUNK_SIZE_PARENT,
                chunk_overlap=cfg.CHUNK_OVERLAP,
                length_function=len,
                )
            
            child_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 2, "lambda_mult": 0.5},
            )
            store = InMemoryStore()
            self.retriever =  ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            retriever = child_retriever
            )
            
            self.retriever.add_documents(documents, ids=None)
                
            
        except Exception as e:
            print(f"Errore durante l'inizializzazione del retriever: {e}")
            exit(1)

    
    
    def set_conversation_history(self,question:str,answer:str):
        """
        Set the conversation history
        """
        self.conversation_history.append({"question":question,"answer":answer})
    


    def retrieve_context(self,question:str):
        """
        Retrieve the context for the question
        """
        print("---RETRIEVE---")
        self.curr_question = question
        retrieved_docs = self.retriever.invoke(question)
        self.context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    
    def generate_answer(self):

        print("---GENERATE---")
        question = self.curr_question
        context = self.context
        prompt = ChatPromptTemplate.from_template(
            """
            Rispondi alla seguente domanda basandoti sulle informazioni fornite nel contesto.
            Se l'informazione non è presente nel contesto, rispondi che non hai abbastanza informazioni.
            Se l'informazione risulta parziale o ambigua, esponi ciò che hai capito in due righe massimo.
            
            Contesto:
            {context}
            
            Domanda: {question}
            
            Risposta:
            """
        )

        qa_chain = (
            {
                "context": lambda x :context,
                "question": lambda x :question,
            }
            | prompt
            |self.llm
            |StrOutputParser()
            )
        try:
            print("Chiamata ad LLM \n\n")
            time.sleep(2) ## rate limits
            self.answer = qa_chain.invoke(question)
            self.set_conversation_history(question,self.answer)
            return self.answer
        except Exception as e:
            print(f"Errore durante la generazione della risposta: {e}")
            exit(1)
        
    def file_save(self):
            """
            Salva la risposta in un file
            """
            print("---COMPILE---")
            new_row = pd.DataFrame({
                "ID": [self.question_id],
                "question": [self.curr_question],
                "response": [self.answer]
            })
            try:
                existing_df = pd.read_csv(cfg.RESPONSES_PATH)
                updated_df = pd.concat([existing_df, new_row], ignore_index=True)
            except FileNotFoundError:
                updated_df = new_row
            updated_df.to_csv(cfg.RESPONSES_PATH, index=False)
            print(f"Risposta salvata per la domanda: {self.question_id}")
            
            