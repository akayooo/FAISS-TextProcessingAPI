# FAISS Text Processing API  

![License](https://img.shields.io/badge/license-MIT-blue.svg)  
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)  

The FAISS Text Processing API is a powerful tool for text processing using FAISS, Sentence Transformers, and LangChain technologies. This project provides a RESTful API for creating indexes, storing and retrieving information, and generating answers to questions.  

## 📋 Key Features  

- **FAISS Index Creation:** Automatic text splitting and embedding generation.  
- **Natural Language Processing:** Use of generative models (e.g., GPT-Neo, FLAN-T5).  
- **RESTful API:** Convenient interaction with the system via HTTP requests.  
- **Multilingual Support:** Works with English and Russian languages.  
- **Image-to-Text Conversion:** Integration with Tesseract for text extraction and BLIP for generating image descriptions with subsequent translation into Russian.

## 🛠 Requirements  

- Python 3.9 or newer  
- Installed libraries from requirements.txt  

## 📦 Installation  

1. Clone the repository:  
   
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Create and activate a virtual environment:  
   
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/MacOS
   venv\Scripts\activate     # For Windows
   ```

3. Install dependencies:  
   
   ```bash
   pip install -r requirements.txt
   ```

4. Create a .env file and add your Hugging Face token:  
   
   ```env
   TOKEN=your_huggingface_api_token
   ```

## 🚀 Running  

1. Start the FastAPI server:  
   
   ```bash
   python api.py
   ```

2. Open your browser and go to [http://127.0.0.1:8000](http://127.0.0.1:8000) to check the API's functionality.  

## 🗂 Endpoints  

### 1. API Status Check  
**GET /**  
Returns a message indicating that the API is working.  

**Example response:**  

```json
{
    "message": "API is working"
}
```

### 2. Create FAISS Index  
**POST /create_faiss_index**  
Creates a FAISS index based on the uploaded file.  

**Parameters:**  
- model_name (string) – Name of the model to be used.  
- chat_id (string) – Chat identifier.  
- file (file) – Text file for processing.  

**Example request:**  

```bash
curl -X POST "http://127.0.0.1:8000/create_faiss_index" \
     -F "model_name=GPT-Neo" \
     -F "chat_id=test_chat" \
     -F "file=@example.txt"
```

### 3. Get Answer to a Question  
**POST /answering**  
Answers a question using the created FAISS index.  

**Parameters:**  
- model_name (string) – Name of the model to be used.  
- chat_id (string) – Chat identifier.  
- question (string) – Question to be answered.  

**Example request:**  

```bash
curl -X POST "http://127.0.0.1:8000/answering" \
     -F "model_name=GPT-Neo" \
     -F "chat_id=test_chat" \
     -F "question=What is FAISS?"
```

**Example response:**  

```json
{
    "answer": "FAISS is a library for efficient similarity search and clustering of dense vectors."
}
```

## 🖼 Image-to-Text Conversion  

This project also includes a method for converting images to text using OCR (Optical Character Recognition) and generating image descriptions using BLIP. Additionally, the image description is translated from English to Russian using a translation model.

### Main Steps of Image Processing:

1. **Text Extraction from Image (OCR):** Using Tesseract, text is extracted from the image in Russian and English.  
2. **Image Description Generation:** The BLIP model generates an image description in English.  
3. **Translation into Russian:** Using the translation model (Helsinki-NLP/opus-mt-en-ru), the image description is translated into Russian.  
4. **Saving Results:** All results are saved in a text file.

### Example Usage:

```
# Example usage
# Path to the uploaded image (replace with the actual path)
uploaded_image_path = "/home/akayo/Downloads/Sail.png"  # Specify the file path

# Call the main function
main(uploaded_image_path)
```

## 📚 License  

This project is licensed under the [MIT](LICENSE) license.

## 📧 **Contact**

For questions or feedback, feel free to reach out:

- **Email**: nikita26.08.98nesterov@gmail.com
- **GitHub**: [AKAYO](https://github.com/akayooo)

---

Happy modeling! 🎉
