from pathlib import Path
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
import config

# --- Inicializar el modelo de Embeddings ---
def obtener_modelo_embeddings(nombre_modelo=config.EMBEDDING_MODEL):
    # multilingual-e5-large corre localmente en el Space sin consumir API
    modelo_embeddings = SentenceTransformerEmbeddings(
        model_name=nombre_modelo
    )
    print("✓ Modelo de embeddings configurado")
    print(f"  {nombre_modelo}, sin API")
    return modelo_embeddings

# --- Instancia Global de la DB en Memoria ---
# Cargamos los embeddings y creamos la base en memoria (sin persist_directory) para HF Spaces
embeddings_global = obtener_modelo_embeddings()

vectorstore = Chroma(
    collection_name=config.COLLECTION_NAME,
    embedding_function=embeddings_global
)

# --- Núcleo de carga: recibe rutas a PDFs (str o Path) ---
def cargar_y_fragmentar_pdfs(rutas_pdfs):
    """Carga documentos PDF a partir de una lista de rutas, los procesa con
    PyPDFLoader y los fragmenta. Es el núcleo común que usan tanto la carga
    automática de la carpeta de datos como la subida manual desde Gradio.
    """
    if not rutas_pdfs:
        return [], []

    documentos_crudos = []
    nombres_archivos = []

    for ruta in rutas_pdfs:
        ruta = Path(ruta)
        nombre_corto = ruta.name
        nombres_archivos.append(nombre_corto)

        loader = PyPDFLoader(str(ruta))
        paginas = loader.load()
        documentos_crudos.extend(paginas)
        print(f"  ✓ {nombre_corto} — {len(paginas)} páginas")

    print(f"\n✓ Total: {len(documentos_crudos)} páginas cargadas")

    divisor = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.SEPARATORS
    )

    fragmentos = divisor.split_documents(documentos_crudos)
    print(f"✓ Fragmentación completada")
    print(f"  Documentos originales: {len(documentos_crudos)} páginas")
    print(f"  Fragmentos generados:  {len(fragmentos)}")

    if fragmentos:
        print(f"\n◈ Ejemplo de fragmento:")
        print(f"  Fuente: {Path(fragmentos[0].metadata.get('source', 'desconocida')).name}")
        print(f"  Página: {fragmentos[0].metadata.get('page', '?')}")
        print(f"  Texto:  {fragmentos[0].page_content[:200]}...")

    return nombres_archivos, fragmentos


def cargar_y_fragmentar_pdfs_gradio(archivos_gradio):
    """Wrapper para cuando los PDFs vienen subidos desde la interfaz de Gradio
    (objetos de archivo temporal, no rutas simples)."""
    if not archivos_gradio:
        return [], []
    rutas = [archivo.name for archivo in archivos_gradio]
    return cargar_y_fragmentar_pdfs(rutas)


def guardar_en_vector_db_gradio(fragmentos: list):
    """Toma los fragmentos y los indexa en la instancia activa de ChromaDB."""
    cantidad_fragmentos = len(fragmentos)
    print(f"\n💾 Guardando {cantidad_fragmentos} fragmentos en ChromaDB (Memoria)...")

    vectorstore.add_documents(fragmentos)

    print(f"✓ Base vectorial lista")
    print(f"  Colección: {config.COLLECTION_NAME}")
    print(f"  Fragmentos indexados en este lote: {cantidad_fragmentos}")
    print("✓ Base vectorial actualizada en memoria para el Space.")


# --- La función orquestadora que llama app.py desde la pestaña de Gradio ---
def cargar_pdfs_interfaz(archivos):
    """Función que unifica la carga y el guardado para Gradio."""
    print("=== INICIANDO PIPELINE DE INGESTA DESDE GRADIO ===")

    nombres, fragmentos = cargar_y_fragmentar_pdfs_gradio(archivos)
    if not fragmentos:
        return "❌ Proceso abortado: No se generaron fragmentos."

    guardar_en_vector_db_gradio(fragmentos)

    return f"✓ Archivos: {', '.join(nombres)}\n✓ Fragmentos indexados: {len(fragmentos)}"


# --- Carga automática de la carpeta de datos al iniciar la app ---
def indexar_carpeta_datos():
    """Indexa automáticamente todos los PDFs de config.CARPETA_DATOS al
    arrancar la app, para que el Space tenga conocimiento base sin que
    nadie tenga que subir nada manualmente. Se ejecuta una sola vez,
    al importar este módulo.
    """
    carpeta = config.CARPETA_DATOS

    if not carpeta.exists():
        print(f"⚠ No existe la carpeta '{carpeta.name}/'; se omite la carga inicial.")
        return 0, []

    rutas_pdfs = sorted(carpeta.glob("*.pdf"))
    if not rutas_pdfs:
        print(f"⚠ No hay PDFs en '{carpeta.name}/'; se omite la carga inicial.")
        return 0, []

    print(f"\n📂 Indexando {len(rutas_pdfs)} PDF(s) precargados desde '{carpeta.name}/'...")
    nombres, fragmentos = cargar_y_fragmentar_pdfs(rutas_pdfs)
    if fragmentos:
        guardar_en_vector_db_gradio(fragmentos)

    return len(fragmentos), nombres


FRAGMENTOS_INICIALES, ARCHIVOS_INICIALES = indexar_carpeta_datos()