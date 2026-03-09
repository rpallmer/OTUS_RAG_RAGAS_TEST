# test_rag_chain_evaluation.py

import pytest
from rag_qdrant_ragas import rag_chain, llm_ragas # Импортируем цепочку и оценочную модель из вашего основного файла
from ragas_fun import evaluate_rag_response # Импортируем функцию оценки

# Тестовые данные
TEST_QUERY = "Я переехал, хочу расторгнуть договор с вашей компанией. Что для этого нужно сделать?"
TEST_REFERENCE = """На Ваше обращение о переоформлении лицевого счета/ о расторжении договора ресурсоснабжения сообщаем.
С учетом положений ч. 1 ст. 452, ч. 1 ст. 546 Гражданского кодекса Российской Федерации, ч. 2 ст. 153 Жилищного кодекса Российской Федерации, для переоформления лицевого счета на принадлежавшее ранее Вам (находившееся в пользовании) жилое помещение, необходимо представить следующие документы: 
1. Заявление о расторжении договора энергоснабжения/ на переоформление лицевого счета, содержащее следующую информацию:
• фамилия, имя, отчество;
• реквизиты документа, удостоверяющего личность;
• адрес жилого помещения в многоквартирном доме, по которому предоставляется коммунальная услуга.
2. Документ, подтверждающий переход права собственности (владения, пользования) на помещение в многоквартирном/частном доме (Выписка из Единого государственного реестра, акт приема-передачи помещения и др.).
3. Паспорт гражданина Российской Федерации или иной документ, удостоверяющий личность заявителя.
4. В случае оборудования жилого помещения индивидуальными, общими (квартирными), комнатными приборами учета коммунальных ресурсов рекомендуем Вам в акте приема-передачи помещения указать показания приборов учета, чтобы избежать образование задолженности продавца на момент передачи недвижимости.
Обращаем внимание, что Вы вправе расторгнуть договор энергоснабжения в одностороннем порядке при условии полной оплаты использованной энергии.

Предоставить вышеуказанные документы Вы можете любым удобным для Вас способом:
1. Подать Заявку, заполнив поля формы "Заключить договор онлайн" и приложив необходимые документы на нашем сайте www.esplus.ru в разделе онлайн-сервисы.
2. Обратиться в офисы обслуживания Клиентов компании АО «ЭнергосбыТ Плюс». Полный перечень офисов и режим работы представлен на сайте www.esplus.ru."""

@pytest.mark.evaluation
def test_rag_chain_metrics_thresholds():
    """
    Тестирует цепочку rag_chain и проверяет,
    что метрики Ragas превышают минимальные пороги.
    """
    # Пороги
    MIN_FAITHFULNESS = 0.7
    MIN_FACTUAL_CORRECTNESS = 0.7
    MIN_CONTEXT_RECALL = 0.5

    # Вызов цепочки для получения ответа и извлеченных документов
    # rag_chain ожидает словарь с ключами "question" и "reference"
    chain_output = rag_chain.invoke({"question": TEST_QUERY, "reference": TEST_REFERENCE})

    # Извлечение результатов из выходных данных цепочки
    # Предполагаем, что цепочка возвращает словарь, содержащий как минимум "response", "question", "reference", "context_docs_and_scores"
    TEST_RESPONSE = chain_output.get("response", "")
    # TEST_RELEVANT_DOCS - это список строк (Answer из metadata), который уже подготовлен внутри run_evaluation
    # Чтобы получить его для теста, нужно повторить логику из run_evaluation или изменить rag_chain,
    # чтобы она возвращала и этот список. Для простоты, повторим логику здесь.
    context_docs_and_scores = chain_output.get("context_docs_and_scores", [])
    TEST_RELEVANT_DOCS_STRINGS = [doc.metadata.get('Answer', '') for doc, score in context_docs_and_scores]

    print(f"Тестовый вопрос: {TEST_QUERY}")
    print(f"Сгенерированный ответ: {TEST_RESPONSE}")
    print(f"Извлеченные документы (Answer): {TEST_RELEVANT_DOCS_STRINGS}")
    print(f"Эталонный ответ: {TEST_REFERENCE}")


    # Вызов функции оценки Ragas с результатами цепочки
    evaluation_result_obj = evaluate_rag_response(
        llm=llm_ragas, # Используем оценочную модель из основного файла
        response=TEST_RESPONSE,
        reference=TEST_REFERENCE, # Используем эталонный ответ из переменной
        relevant_docs=TEST_RELEVANT_DOCS_STRINGS, # Используем документы, извлеченные цепочкой
        query=TEST_QUERY # Используем вопрос из переменной
    )
    
    #print(f"\n--- DEBUG: Тип объекта result_obj ---")
    #print(type(result_obj))
    #print(f"--- DEBUG: Содержимое result_obj ---")
    #print(result_obj)
    #print(f"--- DEBUG: Атрибуты result_obj ---")
    #print(dir(result_obj))
    #if hasattr(result_obj, 'scores'):
    #    print(f"--- DEBUG: Атрибут scores ---")
    #    print(result_obj.scores)
    #    print(f"--- DEBUG: Тип атрибута scores ---")
    #    print(type(result_obj.scores))
    
    evaluation_result = evaluation_result_obj.scores[0]
    print(f"\n--- Результаты оценки RAG ---")
    print(evaluation_result)

    faithfulness_val = evaluation_result.get('faithfulness')
    factual_corr_val = evaluation_result.get('factual_correctness(mode=f1)')
    context_recall_val = evaluation_result.get('context_recall')

    # Проверяем, что значения не являются None или NaN и превышают порог
    assert faithfulness_val is not None and faithfulness_val != float('nan'), \
        f"Faithfulness is {faithfulness_val}, ожидалось число."
    assert faithfulness_val >= MIN_FAITHFULNESS, \
        f"Faithfulness {faithfulness_val} ниже порога {MIN_FAITHFULNESS}. Результат: {evaluation_result}"

    assert factual_corr_val is not None and factual_corr_val != float('nan'), \
        f"FactualCorrectness is {factual_corr_val}, ожидалось число."
    assert factual_corr_val >= MIN_FACTUAL_CORRECTNESS, \
        f"FactualCorrectness {factual_corr_val} ниже порога {MIN_FACTUAL_CORRECTNESS}. Результат: {evaluation_result}"

    assert context_recall_val is not None and context_recall_val != float('nan'), \
        f"ContextRecall is {context_recall_val}, ожидалось число."
    assert context_recall_val >= MIN_CONTEXT_RECALL, \
        f"ContextRecall {context_recall_val} ниже порога {MIN_CONTEXT_RECALL}. Результат: {evaluation_result}"

    print("--- Все метрики превышают пороги ---")

if __name__ == "__main__":
    # Для запуска теста: pytest test_rag_chain_evaluation.py::test_rag_chain_metrics_thresholds -s
    # или просто: pytest test_rag_chain_evaluation.py -s -m evaluation
    pass