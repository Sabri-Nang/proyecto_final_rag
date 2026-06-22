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

# --- Tu función modificada para la interfaz de Gradio ---
def cargar_y_fragmentar_pdfs_gradio(archivos_gradio):
    """Carga los documentos PDF subidos desde la interfaz de Gradio,
    los procesa usando PyPDFLoader, y muestra información de progreso.
    """
    if not archivos_gradio:
        return "No se seleccionaron archivos.", []

    documentos_crudos = []
    nombres_archivos = []

    # En Gradio, 'archivos_gradio' es una lista de objetos de tipo archivo
    for archivo in archivos_gradio:
        nombre_corto = Path(archivo.name).name
        nombres_archivos.append(nombre_corto)
        
        # Cargamos usando la ruta temporal que nos da Gradio (.name)
        loader = PyPDFLoader(archivo.name)
        paginas = loader.load()
        documentos_crudos.extend(paginas)
        print(f"  ✓ {nombre_corto} — {len(paginas)} páginas")

    print(f"\n✓ Total: {len(documentos_crudos)} páginas cargadas desde la interfaz")

    # Tu divisor original con las constantes del config
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.SEPARATORS
    )

    fragmentos = divisor.split_documents(documentos_crudos)
    print(f"✓ Fragmentación completada")
    print(f"  Documentos originales: {len(documentos_crudos)} páginas")
    print(f"  Fragmentos generados:  {len(fragmentos)}")

    # Log de ejemplo idéntico al tuyo
    if fragmentos:
        print(f"\n◈ Ejemplo de fragmento:")
        print(f"  Fuente: {Path(fragmentos[0].metadata.get('source', 'desconocida')).name}")
        print(f"  Página: {fragmentos[0].metadata.get('page', '?')}")
        print(f"  Texto:  {fragmentos[0].page_content[:200]}...")
        
    return nombres_archivos, fragmentos

# --- Tu función de guardado adaptada a la DB en memoria ---
def guardar_en_vector_db_gradio(fragmentos: list):
    """Toma los fragmentos y los indexa en la instancia activa de ChromaDB."""
    print(f"\n💾 Guardando {len(fragmentof := len(fragmentos))} fragmentos en ChromaDB (Memoria)...")

    # Añadimos los documentos a la instancia global
    vectorstore.add_documents(fragmentos)

    print(f"✓ Base vectorial lista")
    print(f"  Colección: {config.COLLECTION_NAME}")
    print(f"  Fragmentos indexados en este lote: {fragmentof}")
    print("✓ Base vectorial actualizada en memoria para el Space.")

# --- La función orquestadora que llamará app.py ---
def cargar_pdfs_interfaz(archivos):
    """Función que unifica la carga y el guardado para Gradio."""
    print("=== INICIANDO PIPELINE DE INGESTA DESDE GRADIO ===")
    
    nombres, fragmentos = cargar_y_fragmentar_pdfs_gradio(archivos)
    if not fragmentos:
        return "❌ Proceso abortado: No se generaron fragmentos."
        
    guardar_en_vector_db_gradio(fragmentos)
    
    return f"✓ Archivos: {', '.join(nombres)}\n✓ Fragmentos indexados: {len(fragmentos)}"