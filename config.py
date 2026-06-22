import os

# Autenticación y API de HuggingFace
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

if not HF_TOKEN:
    raise ValueError("Configurá el secreto HF_TOKEN en el Space.")

# Configuración de Embeddings y Base Vectorial
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
COLLECTION_NAME = "proyecto_rag_spaces"

# Parámetros del Divisor de Texto
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
SEPARATORS = ["\n\n", "\n", ". ", " "]