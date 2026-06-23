import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# 🎛️ CONFIGURACIÓN PRINCIPAL: ¿Estás en tu computadora o en Hugging Face?
# ==============================================================================
# Hugging Face Spaces define automáticamente la variable de
# entorno SPACE_ID. Si existe, asumimos que NO estamos en modo local.
# Esto evita el típico olvido de "subí a HF con MODO_LOCAL = True". Si alguna
# vez necesitás forzarlo a mano (por ejemplo para pruebas), seteá la variable
# de entorno MODO_LOCAL_FORZADO=1 (local) o MODO_LOCAL_FORZADO=0 (nube).
_forzado = os.environ.get("MODO_LOCAL_FORZADO")
if _forzado is not None:
    MODO_LOCAL = _forzado == "1"
else:
    MODO_LOCAL = os.environ.get("SPACE_ID") is None
# ==============================================================================

# Autenticación y API de HuggingFace
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Carpeta con los PDFs que se indexan automáticamente al iniciar la app.
CARPETA_DATOS = Path(__file__).resolve().parent / "documentos"

# Configuración según el entorno
if MODO_LOCAL:
    MODEL_NAME = "gemma4:e2b"
    NUM_CTX = 2048
else:
    # Parámetros para Hugging Face
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

# Cantidad de fragmentos que devuelve el retriever por pregunta.
TOP_K = 4