# FAISS Text Processing API  

![License](https://img.shields.io/badge/license-MIT-blue.svg)  
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)  

FAISS Text Processing API – это мощный инструмент для обработки текстов с использованием технологий FAISS, Sentence Transformers и LangChain. Этот проект предоставляет RESTful API для создания индексов, хранения и извлечения информации, а также генерации ответов на основе вопросов.  

## 📋 Основные возможности  

- **Создание FAISS индексов:** автоматическое разбиение текста и генерация эмбеддингов.  
- **Обработка естественного языка:** использование генеративных моделей (например, GPT-Neo, FLAN-T5).  
- **RESTful API:** удобное взаимодействие с системой через HTTP-запросы.  
- **Поддержка многоязычности:** работа с английским и русским языками.  

## 🛠 Требования  

- Python 3.9 или новее  
- Установленные библиотеки из `requirements.txt`  

## 📦 Установка  

1. Клонируйте репозиторий:  
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```  

2. Создайте и активируйте виртуальное окружение:  
   ```bash
   python -m venv venv
   source venv/bin/activate  # Для Linux/MacOS
   venv\Scripts\activate     # Для Windows
   ```  

3. Установите зависимости:  
   ```bash
   pip install -r requirements.txt
   ```  

4. Создайте `.env` файл и добавьте ваш Hugging Face токен:  
   ```env
   TOKEN=your_huggingface_api_token
   ```  

## 🚀 Запуск  

1. Запустите FastAPI сервер:  
   ```bash
   python api.py
   ```  

2. Откройте браузер и перейдите на [http://127.0.0.1:8000](http://127.0.0.1:8000) для проверки работы API.  

## 🗂 Эндпоинты  

### 1. Проверка статуса API  
**GET /**  
Возвращает сообщение о том, что API работает.  

**Пример ответа:**  
```json
{
    "message": "API работает"
}
```  

### 2. Создание FAISS индекса  
**POST /create_faiss_index**  
Создает FAISS индекс на основе загруженного файла.  

**Параметры:**  
- `model_name` (строка) – Название используемой модели.  
- `chat_id` (строка) – Идентификатор чата.  
- `file` (файл) – Текстовый файл для обработки.  

**Пример запроса:**  
```bash
curl -X POST "http://127.0.0.1:8000/create_faiss_index" \
     -F "model_name=GPT-Neo" \
     -F "chat_id=test_chat" \
     -F "file=@example.txt"
```  

### 3. Получение ответа на вопрос  
**POST /answering**  
Отвечает на вопрос с использованием созданного FAISS индекса.  

**Параметры:**  
- `model_name` (строка) – Название используемой модели.  
- `chat_id` (строка) – Идентификатор чата.  
- `question` (строка) – Вопрос, на который нужно ответить.  

**Пример запроса:**  
```bash
curl -X POST "http://127.0.0.1:8000/answering" \
     -F "model_name=GPT-Neo" \
     -F "chat_id=test_chat" \
     -F "question=What is FAISS?"
```  

**Пример ответа:**  
```json
{
    "answer": "FAISS is a library for efficient similarity search and clustering of dense vectors."
}
```  

## 📚 Лицензия  

Этот проект распространяется под лицензией [MIT](LICENSE).  
```
