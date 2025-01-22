import logging
import requests # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
from langchain.embeddings.base import Embeddings # type: ignore
import time
import model_selection
from faiss_init import ParaphraseMpnetBaseV2Embeddings, SbertLargeNluRuEmbeddings
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Загрузка переменной из виртуального окружения
TOKEN = os.getenv("TOKEN")

# Используем переменную TOKEN в headers
headers = {"Authorization": f"Bearer {TOKEN}"}


def main(model, question, chat_id, system_prompt="Ответ должен быть развернутым и содержать 3-4 предложения."):
    try:
        model_name, embedding_model_name, delay = model_selection.model_name_configuration(model)

        API_URL = f'https://api-inference.huggingface.co/models/{model_name}'

        # Загрузка переменной из виртуального окружения
        TOKEN = os.getenv("TOKEN")

        # Используем переменную TOKEN в headers
        headers = {"Authorization": f"Bearer {TOKEN}"}

        # Выбор модели эмбеддингов
        if embedding_model_name == "sberbank-ai/sbert_large_nlu_ru":
            embedding_model = SbertLargeNluRuEmbeddings()
        else:
            embedding_model = ParaphraseMpnetBaseV2Embeddings()

        def query_huggingface_api(system_prompt, context, question, delay):
            """
            Отправка запроса к API Hugging Face для генеративной задачи.
            """
            try:
                combined_prompt = f"{system_prompt}\n\nКонтекст:\n{context}\n\nВопрос: {question}\n\nОтвет:"
                payload = {
                    "inputs": combined_prompt,
                    "parameters": {
                        "max_length": 150,
                        "temperature": 0.6,
                        "top_p": 0.9,
                        "do_sample": True
                    }
                }

                logging.info("Ожидание %s секунд перед отправкой запроса...", delay)
                time.sleep(delay)

                logging.info("Отправка запроса к API Hugging Face...")
                response = requests.post(API_URL, headers=headers, json=payload)
                logging.info("HTTP статус: %s", response.status_code)

                if response.status_code != 200:
                    raise ValueError(f"API Error {response.status_code}: {response.text}")

                full_response = response.json()
                logging.info("Ответ от API: %s", full_response)

                if isinstance(full_response, dict) and "generated_text" in full_response:
                    return full_response["generated_text"]
                elif isinstance(full_response, list) and len(full_response) > 0 and "generated_text" in full_response[0]:
                    return full_response[0]["generated_text"]
                else:
                    raise ValueError("Ответ от API не содержит ключа 'generated_text'.")
            except Exception as e:
                logging.error("Ошибка при запросе к API: %s", e)
                raise

        def faiss_founder(chat_id: str, embedding_model):
            try:
                faiss_index_path = f'faiss/faiss_{chat_id}'
                logging.info("Попытка загрузить существующий индекс...")
                vectorstore = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
                logging.info("Индекс загружен.")
                retriever = vectorstore.as_retriever()
                return retriever
            except Exception as e:
                logging.error("Ошибка при загрузке FAISS индекса: %s", e)
                raise

        def rag_fusion_pipeline(question, chat_id: str, max_context_tokens=150):
            """
            Извлечение релевантного контекста из базы знаний.
            """
            try:
                retriever = faiss_founder(chat_id, embedding_model)
                relevant_documents = retriever.invoke(question, k=3)
                relevant_documents_content = [doc.page_content for doc in relevant_documents]
                combined_context = "\n".join(relevant_documents_content)

                if len(combined_context.split()) > max_context_tokens:
                    combined_context = " ".join(combined_context.split()[:max_context_tokens])

                return combined_context
            except Exception as e:
                logging.error("Ошибка при извлечении контекста: %s", e)
                raise

        def rag_chain(question, chat_id):
            """
            Основная цепочка RAG: извлечение контекста и генерация ответа.
            """
            try:
                context = rag_fusion_pipeline(question, chat_id, max_context_tokens=1024)
                logging.info("Длина контекста в токенах: %s", len(context.split()))
                return query_huggingface_api(system_prompt, context, question, delay)
            except Exception as e:
                logging.error("Ошибка в цепочке RAG: %s", e)
                raise

        def extract_answer_section(text):
            """
            Извлекает текст после "Ответ:" до первого абзаца.
            """
            # Ищем начало фрагмента после слова "Ответ:"
            start_index = text.rfind("Ответ:") + len("Ответ:")
            if start_index == -1:
                return answer_text
            
            # Извлекаем текст после "Ответ:"
            answer_text = text[start_index:].strip()
            
            # Удаляем символы новой строки
            answer_text = answer_text.replace("\n", " ").replace('*', " ").replace('#', " "). replace("  ", ' ').replace(' - ', " ")

            result = answer_text.split("Вопрос:")[0].strip()
            
            return result

        # Основной процесс
        answer = rag_chain(question, chat_id)
        if model == 'GPT-Neo':
            answer = extract_answer_section(answer)

        logging.info("Ответ успешно сформирован.")
        return answer

    except Exception as e:
        logging.critical("Критическая ошибка в main: %s", e)
        raise

'''
# Пример использования
question = "What is Rutube?"
chat_id = "en"
system_prompt = "Just answer in short"
model = 'GPT-Neo'
answer = main(model, question, chat_id, system_prompt=system_prompt)
print("====================================================================")
print(answer)
'''
