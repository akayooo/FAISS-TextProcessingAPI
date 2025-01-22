import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import TextLoader

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ParaphraseMpnetBaseV2Embeddings(Embeddings):
    """
    Адаптер для модели `sentence-transformers/paraphrase-mpnet-base-v2`.
    """
    def __init__(self):
        model_name = "sentence-transformers/paraphrase-mpnet-base-v2"
        logging.info("Инициализация модели эмбеддингов: %s", model_name)
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        logging.info("Выполнение эмбеддинга для %d текстов.", len(texts))
        return self.model.encode(texts, convert_to_tensor=False)

    def embed_query(self, text):
        logging.info("Выполнение эмбеддинга для запроса.")
        return self.embed_documents([text])[0]


class SbertLargeNluRuEmbeddings(Embeddings):
    """
    Адаптер для модели `sberbank-ai/sbert_large_nlu_ru`.
    """
    def __init__(self):
        model_name = "sberbank-ai/sbert_large_nlu_ru"
        logging.info("Инициализация модели эмбеддингов: %s", model_name)
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        logging.info("Выполнение эмбеддинга для %d текстов.", len(texts))
        return self.model.encode(texts, convert_to_tensor=False)

    def embed_query(self, text):
        logging.info("Выполнение эмбеддинга для запроса.")
        return self.embed_documents([text])[0]


def faiss_init_method(embedding_model_name, chat_id, file_path):
    logging.info("Инициализация FAISS с model_name=%s, chat_id=%s, file_path=%s",
                 embedding_model_name, chat_id, file_path)
    try:
        # Загрузка данных
        logging.info("Загрузка данных из файла: %s", file_path)
        text_loader = TextLoader(file_path, encoding='utf-8')
        data = text_loader.load()

        # Выбор модели эмбеддингов
        if embedding_model_name == "sberbank-ai/sbert_large_nlu_ru":
            embeddings = SbertLargeNluRuEmbeddings()
        else:
            embeddings = ParaphraseMpnetBaseV2Embeddings()

        # Разбиение текстов на фрагменты
        logging.info("Разбиение текста на фрагменты...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(data)
        logging.info("Текст успешно разбит на %d фрагментов.", len(texts))

        # Создание векторного хранилища FAISS
        logging.info("Создание FAISS индекса...")
        vectorstore = FAISS.from_documents(texts, embeddings)

        # Сохранение хранилища на диск
        faiss_index_path = f'faiss/faiss_{chat_id}'
        vectorstore.save_local(faiss_index_path)
        logging.info("FAISS индекс успешно создан и сохранен в: %s", faiss_index_path)

    except Exception as e:
        logging.error("Ошибка при инициализации FAISS: %s", e)
        raise
