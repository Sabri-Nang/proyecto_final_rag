import os
import platform
import shutil
import subprocess
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Ignorar warnings de inicialización de embeddings
warnings.filterwarnings("ignore")

# Cargar las variables desde el archivo .env
load_dotenv()

# --- Configuración de Rutas (Mejorado para compatibilidad) ---
BASE_DIR = Path(__file__).resolve().parent
NOMBRE_CARPETA_DATOS = os.getenv("CARPETA_DATOS", "CARPETA_DATOS")
NOMBRE_DIRECTORIO_CHROMA = os.getenv("DIRECTORIO_CHROMA", "chroma_db")

CARPETA_DATOS = BASE_DIR / NOMBRE_CARPETA_DATOS
DIRECTORIO_CHROMA = BASE_DIR / NOMBRE_DIRECTORIO_CHROMA

# --- Configuración de Parámetros RAG Fijos ---
EMBEDDING_MODEL = "sentence-transformers/multilingual-e5-large"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Variables Globales Dinámicas ---
MODEL_NAME = ""
NUM_CTX = 2048
PERFIL_HARDWARE = ""

def _detectar_configuracion_hardware():
    """Determina las capacidades de hardware de forma dinámica."""
    global MODEL_NAME, NUM_CTX, PERFIL_HARDWARE

    env_model = os.getenv("MODEL_NAME")
    env_ctx = os.getenv("NUM_CTX")

    sistema = platform.system()
    maquina = platform.machine()

    tiene_cuda = False
    if sistema in ("Linux", "Windows") and shutil.which("nvidia-smi"):
        try:
            resultado = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=False
            )
            tiene_cuda = resultado.returncode == 0
        except Exception:
            tiene_cuda = False

    es_apple_silicon = sistema == "Darwin" and maquina == "arm64"

    if es_apple_silicon:
        PERFIL_HARDWARE = "Apple Silicon (Metal)"
        default_model = "gemma4:e2b"
        default_ctx = 4096
    elif tiene_cuda:
        PERFIL_HARDWARE = "NVIDIA GPU (CUDA)"
        default_model = "granite4:latest"
        default_ctx = 8192
    else:
        PERFIL_HARDWARE = "CPU (sin GPU dedicada)"
        default_model = "gemma4:e2b"
        default_ctx = 1024

    MODEL_NAME = env_model if env_model else default_model
    NUM_CTX = int(env_ctx) if env_ctx else default_ctx

def inicializar_entorno():
    """Prepara el entorno creando directorios y configurando variables."""
    # Evitamos que cree carpetas automáticamente en Hugging Face si ya existen en la build
    if not CARPETA_DATOS.exists():
        CARPETA_DATOS.mkdir(parents=True, exist_ok=True)
    if not DIRECTORIO_CHROMA.exists():
        DIRECTORIO_CHROMA.mkdir(parents=True, exist_ok=True)

    _detectar_configuracion_hardware()

# Ejecutar la inicialización al importar para que query.py e indexer.py tengan las variables listas
inicializar_entorno()