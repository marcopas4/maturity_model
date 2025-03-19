import agent

def compile(agent:agent.Agent):
    '''Funzione per compilare le domande del maturity model. Viene passato l'agente come oggetto per poter utilizzare i suoi metodi'''

    questions = agent.questions

    for idx, question in enumerate(questions):
        #Setto l'ID della domanda
        agent.question_id = idx+1
        #Recupero il contesto
        agent.retrieve_context(question)
        #Genero la risposta
        agent.generate_answer()
        #Salvo la risposta
        agent.file_save()