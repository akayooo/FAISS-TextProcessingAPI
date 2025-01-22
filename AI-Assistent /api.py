import logging
from fastapi import FastAPI, UploadFile, Form, HTTPException # type: ignore
from pathlib import Path
import uvicorn # type: ignore
import os
import faiss_init 
import model
import model_selection


# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Загрузка переменной из виртуального окружения
TOKEN = os.getenv("TOKEN")

# Используем переменную TOKEN в headers
headers = {"Authorization": f"Bearer {TOKEN}"}

# Инициализация FastAPI
app = FastAPI(
    title="FAISS Text Processing API",
    description="API для обработки текстовых файлов с помощью FAISS",
    version="1.0.0"
)


@app.get("/")
def start():
    logging.info("Эндпоинт '/' был вызван.")
    return {"message": "API работает"}


@app.post("/create_faiss_index")
async def create_faiss_index(
    model_name: str = Form(...),
    chat_id: str = Form(...),
    file: UploadFile = Form(...)
):
    """
    Эндпоинт для создания FAISS индекса с использованием faiss_init.
    """
    logging.info("Получен запрос на создание FAISS индекса.")
    logging.info("Параметры: model_name=%s, chat_id=%s", model_name, chat_id)

    try:
        _, embedding_model_name, _ = model_selection.model_name_configuration(model_name)
        logging.info("Используемая модель эмбеддингов: %s", embedding_model_name)

        # Создание временной папки для файла
        temp_dir = Path("temp_files")
        temp_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Временная папка создана: %s", temp_dir)

        # Путь для сохранения временного файла
        file_path = temp_dir / file.filename

        # Сохранение файла
        file_content = await file.read()
        if not file_content.strip():
            logging.error("Файл пуст.")
            raise HTTPException(status_code=400, detail="Файл пуст")

        with open(file_path, "wb") as temp_file:
            temp_file.write(file_content)
        logging.info("Файл сохранен: %s", file_path)

        # Вызов faiss_init с временным файлом
        faiss_init.faiss_init_method(embedding_model_name, chat_id, str(file_path))
        logging.info("FAISS индекс успешно создан для chat_id=%s", chat_id)

        # Удаление временного файла
        os.remove(file_path)
        logging.info("Временный файл удален: %s", file_path)

        return {"message": f"FAISS индекс успешно создан для chat_id {chat_id}!"}
    
    except Exception as e:
        logging.error("Ошибка при создании FAISS индекса: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка при создании FAISS индекса: {str(e)}")


@app.post("/answering")
async def answering(model_name: str = Form(...),
                    chat_id: str = Form(...),
                    question: str = Form(...)):
    """ 
    Эндпоинт для ответа на вопрос
    """
    logging.info("Получен запрос на ответ: вопрос='%s', chat_id='%s', model_name='%s'", question, chat_id, model_name)

    try:
        answer = model.main(model_name, question, chat_id)
        logging.info("Ответ успешно сформирован: %s", answer)
        return {"answer": answer}
    except Exception as e:
        logging.error("Ошибка при формировании ответа: %s", e)
        return {"error": str(e)}


# Запуск приложения
if __name__ == "__main__":
    logging.info("Запуск приложения...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
