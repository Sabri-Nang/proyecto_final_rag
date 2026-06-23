import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# 🎛️ CONFIGURACIÓN PRINCIPAL: ¿Estás en tu computadora o en Hugging Face?
# ==============================================================================
MODO_LOCAL = True  # Poner en False para subir a Hugging Face Spaces 🚀
# ==============================================================================

# Autenticación y API de HuggingFace
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Configuración según el entorno
if MODO_LOCAL:
    MODEL_NAME = "gemma4:e2b"
    NUM_CTX = 2048
else:
    # PArámetros para Hugging Face
    MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
    if not HF_TOKEN:
        raise ValueError("Configurá el secreto HF_TOKEN en el Space.")

# Configuración de Embeddings y Base Vectorial
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
COLLECTION_NAME = "proyecto_rag_spaces"

# Parámetros del Divisor de Texto
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ". ", " "]