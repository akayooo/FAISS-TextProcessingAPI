# nbasilico/euro-llm-9b" - попробовать EuroLLM_9B

def model_name_configuration(model: str):
    """
    Конфигурирует модель на основе переданного названия.
    
    Args:
        model (str): Название модели.

    Returns:
        Tuple[str, str, int]: Название модели, название модели эмбеддингов, задержка.

    Raises:
        ValueError: Если передано неизвестное название модели.
    """
    match model:
        # Английский язык
        case 'GPT-Neo':
            model_name = "EleutherAI/gpt-neo-2.7B"  # Генеративная модель
            embedding_model_name = "sentence-transformers/paraphrase-mpnet-base-v2"  # Эмбеддинги для английского
            delay = 1

        case 'FLAN-T5':
            model_name = "google/flan-t5-large"  # Генеративная модель
            embedding_model_name = "sentence-transformers/paraphrase-mpnet-base-v2"  # Эмбеддинги для английского
            delay = 126

        # Русский язык
        case 'RuGPT3Large':
            model_name = "sberbank-ai/rugpt3large_based_on_gpt2"  # Генеративная модель для русского языка
            embedding_model_name = "sberbank-ai/sbert_large_nlu_ru"  # Эмбеддинги для русского языка
            delay = 170

        case 'RuGPT3Medium':
            model_name = "ai-forever/rugpt3medium_based_on_gpt2"  # Более легкая версия RuGPT
            embedding_model_name = "sberbank-ai/sbert_large_nlu_ru"  # Эмбеддинги для русского языка
            delay = 100
            
        # case 'GPT-J':
        #     model_name = "EleutherAI/gpt-j-6B"  # Генеративная модель
        #     embedding_model_name = "sentence-transformers/paraphrase-mpnet-base-v2"  # Эмбеддинги для английского
        #     delay = 1


        # case 'OPT':
        #     model_name = "facebook/opt-1.3b"  # Генеративная модель
        #     embedding_model_name = "sentence-transformers/paraphrase-mpnet-base-v2"  # Эмбеддинги для английского
        #     delay = 120

        # case 'BLOOM':
        #     model_name = "bigscience/bloom"  # Многоязычная генеративная модель
        #     embedding_model_name = "sberbank-ai/sbert_large_nlu_ru"  # Эмбеддинги для русского языка
        #     delay = 1

        case _:
            raise ValueError(f"Неизвестное название модели: {model}. Доступные модели: 'GPT-Neo', 'GPT-J', 'FLAN-T5', 'OPT', 'RuGPT3Large', 'RuGPT3Medium', 'BLOOM'.")

    return model_name, embedding_model_name, delay
