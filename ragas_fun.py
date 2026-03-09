def evaluate_rag_response(llm, response, reference, relevant_docs, query):
    """
    Evaluates a RAG response using ragas metrics
    
    Args:
        model to be used as evaluator
        response: Generated response from RAG system
        reference: Expected/golden response
        relevant_docs: Retrieved documents (list of Document objects) used for generating response
        query: Original user query
        
    Returns:
        Evaluation results dictionary
    """
    from ragas import EvaluationDataset, evaluate
    # from ragas.llms import LangchainLLMWrapper # Этот импорт, возможно, больше не нужен в новых версиях Ragas
    from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
    
    # Преобразуем список Document в список строк (их page_content)
    #retrieved_contexts_str = [doc.page_content for doc in relevant_docs]

    # Create single item dataset
    dataset = [{
        "user_input": query,
        "retrieved_contexts": relevant_docs, # Передаем список строк
        "response": response,
        "reference": reference
    }]
    
    evaluation_dataset = EvaluationDataset.from_list(dataset)
    # evaluator_llm = LangchainLLMWrapper(llm) # Удалите, если используете новую версию Ragas

    # Передаем llm напрямую в evaluate, если он совместим с Langchain
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        llm=llm # Передаем оригинальный llm
    )
    
    return result