import os
import re
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Инструменты RAG
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

#from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Импорты LangChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langfuse.langchain import CallbackHandler
from langchain_openai import ChatOpenAI


# Импорты LangGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

load_dotenv()

from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from ragas.dataset_schema import SingleTurnSample
from ragas import EvaluationDataset
from openai import OpenAI

# Визуализация
import pandas as pd
from IPython.display import display, Markdown

# Конфиг
DATA_DIR = Path("../data/")
COLLECTION_NAME = "client_ragas_docs"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "BAAI/bge-m3"

import torch
#from pre_data import parse_faq_text,read_text_file
from convert_data_for_chanck import load_and_enrich_documents
from ragas_fun import evaluate_rag_response

from langchain_community.embeddings import OllamaEmbeddings
# настройка для работы судьи
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# подключение landfuse для мониторинга
try:
        langfuse_handler = CallbackHandler()
        print("✅ Langfuse мониторинг подключен")
except Exception as e:
        print(f"⚠️ Ошибка подключения Langfuse: {e}")
        langfuse_handler = None
    # Добавляем его в конфиг
callbacks: list[BaseCallbackHandler] = [langfuse_handler] if langfuse_handler else []
config: RunnableConfig = {
        "configurable": {"thread_id": "session_1"},
        "callbacks": callbacks,
    }

embeddings = OllamaEmbeddings(
model="bge-m3",
base_url="http://localhost:11434",
)
# проверка  моделм эмбедингов
print(f"⏳ подключение к модели эмбеддингов {EMBEDDING_MODEL}")
test_vec = embeddings.embed_query("Тестовый запрос")
print(len(test_vec))
# подключение к векторной базе развернутой в Docker
client = QdrantClient(url=QDRANT_URL)
# Проверяем, есть ли коллекция, и пересоздаем её для чистоты эксперимента
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
    print(f"🗑️ Удалена старая коллекция {COLLECTION_NAME}")
print(f"🛠 Создание коллекции с HNSW индексом...")
client.create_collection(
    collection_name=COLLECTION_NAME,
    # размер вектора 1024 определяется размером вектора embadding модели
    # используем Distance.COSINE оптимально для текста, когда важен  угол а не абсолютное значение вектора
    # как при параметре Distance.EUCLID
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    # Настройка HNSW
    hnsw_config=models.HnswConfigDiff(
        m=16,               # Количество связей на узел (больше = точнее, но больше памяти)
        ef_construct=100    # Глубина поиска при построении индекса
    )
)
print("✅ Коллекция готова!")
print("⏳ Запуск индексации документов (может занять время)...")
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)
file_path = "data/Шаблоны_ответов_ для_ИИ.txt"
docs=load_and_enrich_documents(file_path)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,      # Указываем размер с запасом, фактически текст будет меньше и разделять символами разделения
    chunk_overlap=200,    # Нахлест, чтобы не терялся контекст на границах
    separators=["\n\n", "\n## ",  "\n", " ", ""] # Приоритет разделителей (два перевода строки)
)
splits = text_splitter.split_documents(docs)
print(f"📚 Исходных документов: {len(docs)}")
print(f"✂️  Получено чанков: {len(splits)}")
# вывод на печать, получившихся чанков для тестирования
#for entry in splits:
#    print(f"Вопросы: {entry.page_content}")      # ✅
#    print(f"Категория: {entry.metadata['category']}")      # ✅
#    print(f"Субкатегория: {entry.metadata['sub_category']}") # ✅
#    print(f"Ответ: {entry.metadata['Answer']}")      # ✅
#    print("-" * 50)
#   Добавление в векторную базу чанков документов
vector_store.add_documents(splits)
print(f"🎉 Успешно проиндексировано {len(splits)} чанков.")


#openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_REY_RAGAS"))
#llm_ragas= llm_factory("gpt-4o", client=openai_client)

openai_api_key = os.getenv("OPENAI_API_REY_RAGAS")

# Инициализируем модель, передав API-ключ
llm_ragas = ChatOpenAI(
    model="gpt-4o", 
    api_key=openai_api_key,
    callbacks=callbacks
    )

# собираем цепочку
# 1. Инициализация LLM
llm = ChatOllama(
    model="mistral:7b",
    temperature=0.1, # Низкая температура для фактологической точности
    base_url="http://localhost:11434",
    callbacks=callbacks,  # <-- добавляем callback
    timeout=12000
)

#llm=llm_ragas

# 2. Промпт (Инструкция)
# Используем шаблон, который заставляет модель опираться ТОЛЬКО на контекст.
template = """Ты — корпоративный ассистент компании ЭнергосбыТ Плюс".
Ответь на вопрос клиента, используя ТОЛЬКО предоставленный ниже контекст.
Если в контексте нет информации, скажи "В документах нет информации об этом".
Не придумывай факты.
Контекст:
{context}
Вопрос: {question}
Ответ:"""
prompt = ChatPromptTemplate.from_template(template)

# 3. Функция форматирования документов в строку
def format_docs_score(docs_and_scores):
    # docs_and_scores — это список вида [(doc, score), ...]
    formatted_parts = []
    for i, (doc, score) in enumerate(docs_and_scores):
        # Получаем текст ответа из metadata
        answer = doc.metadata.get("Answer", "")
        # Получаем page_content
        content = doc.page_content
        # Формируем строку для текущего документа
        formatted_part = f"[Документ {i+1}] Score: {score:.4f}\n[Answer: {answer}]\n{content}"
        formatted_parts.append(formatted_part)
    # Соединяем все части, разделяя двумя переносами строки
    return "\n".join(formatted_parts)
# 4. Ретривер использует метод as_retriever (не возвращает оценку Score)
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.66  # добавляем порог схожсти для отсеивания отобранных докментов
    }
)
# 4.1 Ретривер использует метод imilarity_search_with_score ( возвращает оценку Score)
def custom_retrieve(query: str):
    results = vector_store.similarity_search_with_score(query, k=3)
    return results

retriever_lambda = RunnableLambda(custom_retrieve)
# 5. вывод результата поиска (переменная context)на экран
def print_context(inputs):
    print("Context:\n", inputs["context"])  # выводим context
    return inputs  # возвращаем тот же словарь
# 5.1 вывод результата поиска c оценкой + переменная context на экран
def print_context_with_scores(inputs):
    context_docs_and_scores = inputs.get("context_docs_and_scores", [])
    print("\n--- Документы и их оценки (scores) ---")
    for i, (doc, score) in enumerate(context_docs_and_scores):
        Answer = doc.metadata.get('Answer', 'N/A')
        category = doc.metadata.get('category', 'N/A')
        print(f"[{i+1}] Score: {score:.4f} | Ответ: {Answer} | Категория: {category}")
        print(f"    Текст: {doc.page_content[:100]}...\n")
    # print("Context (форматированный):\n", inputs["context"])
    return inputs

# Функция для вызова evaluate_rag_response
def run_evaluation(inputs):

    question_str = str(inputs.get('question'))
    reference_str = str(inputs.get('reference'))
    response_str = str(inputs.get('response'))
                
    context_docs_and_scores = inputs.get("context_docs_and_scores", [])
    # Извлекаем только Document из пар (doc, score)
    # only_documents_list = [doc.page_content for doc, score in context_docs_and_scores] if context_docs_and_scores else [] 
    only_documents_list = [doc.metadata.get('Answer', '') for doc, score in  context_docs_and_scores]

    print(f"Type of inputs: {type(inputs)}")
    print(f"Content of inputs: {inputs}")


        # Выполняем оценку
    evaluation_result = evaluate_rag_response(
        llm=llm_ragas,
        response=response_str,
        reference=reference_str,
        relevant_docs=only_documents_list,
        query=question_str
    )
        
    # Выводим результаты оценки (опционально)
    print("\n--- Результаты оценки RAG ---")
    print(evaluation_result)
    
    return inputs  # Возвращаем те же входные данные для продолжения цепочки

# 6. Сборка цепочки (LCEL - LangChain Expression Language)
generation_chain = prompt | llm | StrOutputParser()

rag_chain = (
    {
        "context_docs_and_scores": retriever_lambda, # [(doc, score), ...] -> список
        "context": retriever_lambda | format_docs_score, # строка
        "question": RunnablePassthrough(), # строка
        "reference": RunnablePassthrough()  # строка
    }
    | RunnableLambda(print_context_with_scores) # передает тот же словарь
    # Теперь объединяем результаты генерации с остальными данными
    | {
        "response": lambda x: generation_chain.invoke({"context": x["context"], "question": x["question"]}),
        "question": lambda x: x["question"],
        "context_docs_and_scores": lambda x: x["context_docs_and_scores"],
        "reference": lambda x: x["reference"]
     }
    | RunnableLambda(run_evaluation) # теперь все ключи доступны в inputs
)
print("🔗 RAG Chain собрана!")

if __name__ == "__main__":
    # проверка устройства
    if torch.backends.mps.is_available():
        device = "mps"   # GPU на Mac (Apple Silicon)
    elif torch.cuda.is_available():
        device = "cuda"  # GPU NVIDIA (если вдруг есть)
    else:
        device = "cpu"
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    elif device == "mps":
        print("   GPU: Apple Silicon (M‑series) via MPS")
 

    # Пример вызова цепочки с параметрами question и reference
    question = "Я переехал, хочу расторгнуть договор с вашей компанией. Что для этого нужно сделать? "
    reference = """    На Ваше обращение о переоформлении лицевого счета/ о расторжении договора ресурсоснабжения сообщаем.
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
2. Обратиться в офисы обслуживания Клиентов компании АО «ЭнергосбыТ Плюс». Полный перечень офисов и режим работы представлен на сайте www.esplus.ru." # Пример эталонного ответа
    """
    result = rag_chain.invoke({"question": question, "reference": reference})
    print("\n--- Финальный результат ---")
    print(result)