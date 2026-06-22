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

# --- Configuración de Rutas ---
BASE_DIR = Path.cwd()
NOMBRE_CARPETA_DATOS = os.getenv("CARPETA_DATOS", "CARPETA_DATOS_DEFAULT")
NOMBRE_DIRECTORIO_CHROMA = os.getenv("DIRECTORIO_CHROMA", "chroma_default")

CARPETA_DATOS = BASE_DIR / NOMBRE_CARPETA_DATOS
DIRECTORIO_CHROMA = BASE_DIR / NOMBRE_DIRECTORIO_CHROMA

# --- Inicialización de variables globales de Hardware/Modelo ---
# Las declaramos acá arriba para que sean accesibles mediante 'from config import ...'
MODEL_NAME = ""
NUM_CTX = 2048
PERFIL_HARDWARE = ""


def _detectar_configuracion_hardware():
    """Determina las capacidades de hardware de forma dinámica."""
    global MODEL_NAME, NUM_CTX, PERFIL_HARDWARE

    # 1. Si el usuario ya forzó un modelo en el .env, le damos prioridad
    env_model = os.getenv("MODEL_NAME")
    env_ctx = os.getenv("NUM_CTX")

    sistema = platform.system()
    maquina = platform.machine()

    # Detectar CUDA de forma eficiente
    tiene_cuda = False
    if sistema in ("Linux", "Windows") and shutil.which("nvidia-smi"):
        try:
            resultado = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=False
            )
            tiene_cuda = resultado.returncode == 0
        except Exception:
            tiene_cuda = False

    # 2. Configuración por defecto según arquitectura
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

    # 3. Asignación final (Prioridad: .env > Autodetección)
    MODEL_NAME = env_model if env_model else default_model
    NUM_CTX = int(env_ctx) if env_ctx else default_ctx


def inicializar_entorno():
    """Prepara el entorno local creando directorios y configurando variables globales."""
    print("⚙️  Loading environment variables and verifying paths...")

    # Creamos las carpetas si no existen en el localhost
    CARPETA_DATOS.mkdir(parents=True, exist_ok=True)
    DIRECTORIO_CHROMA.mkdir(parents=True, exist_ok=True)

    print(f" -> Carpeta de PDFs lista en:  {CARPETA_DATOS}")
    print(f" -> Base vectorial lista en:   {DIRECTORIO_CHROMA}\n")

    # Ejecutamos la detección para llenar las variables globales
    _detectar_configuracion_hardware()

    print(f"Sistema:          {platform.system()} — {platform.machine()}")
    print(f"Perfil detectado: {PERFIL_HARDWARE}")
    print(f"Modelo elegido:   {MODEL_NAME}")
    print(f"Contexto (tokens):{NUM_CTX}")
    print("=" * 50 + "\n")


# Se ejecuta automáticamente al importar el módulo por primera vez
inicializar_entorno()