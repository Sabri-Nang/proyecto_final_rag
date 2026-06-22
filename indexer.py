
from config import CARPETA_DATOS, DIRECTORIO_CHROMA
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma

# Listamos todos los PDFs en la carpeta de datos

def listar_pdf(ruta_pdf: str)->bool:
    """Lista la cantidad de pdf en la ruta provista
    
    """
    archivos_pdf = sorted([f for f in os.listdir(CARPETA_DATOS) if f.lower().endswith(".pdf")])

    if not archivos_pdf:
        print("⚠️  No hay PDFs en la carpeta.")
        print(f"   Copiá al menos un PDF en: {ruta_pdf}")
        return False, archivos_pdf
    else:
        print(f"Archivos encontrados: {len(archivos_pdf)}")
        return True, archivos_pdf

def cargar_y_fragmentar_pdfs(ruta_pdf: str):
    """Carga todos los documentos PDF desde el directorio configurado.
    Lista los archivos PDF en CARPETA_DATOS, los carga usando PyPDFLoader,
    y muestra información sobre el número de páginas por archivo y el total.
    
    Args:
        ruta_pdf: Ruta de directorio (parámetro actualmente no utilizado; 
                  la función usa la constante CARPETA_DATOS desde config).
                  
    Note:
        Esta función no devuelve los documentos cargados; solo imprime
        información de progreso en la consola.
    """
    # Y si queremos cargar una carpeta con archivos
    #loader = DirectoryLoader(
    #path="./mi_carpeta",
    #glob="**/*.txt",          # Busca solo TXTs cambiar a .pdf
    #loader_cls=TextLoader,    # Usa el lector de TXTs cambiar a PyPDFLoader para pdf
    #loader_kwargs={"encoding": "utf-8"} # Configuración extra que necesita el TextLoader
#)
    verificacion, archivos_pdf = listar_pdf(ruta_pdf)
    if verificacion:
        documentos_crudos = []

    for nombre_archivo in archivos_pdf:
        ruta_completa = os.path.join(CARPETA_DATOS, nombre_archivo)
        loader = PyPDFLoader(ruta_completa)
        paginas = loader.load()
        documentos_crudos.extend(paginas)
        print(f"  ✓ {nombre_archivo} — {len(paginas)} páginas")

    print(f"\n✓ Total: {len(documentos_crudos)} páginas cargadas")

    # RecursiveCharacterTextSplitter intenta dividir por párrafos primero,
    # luego por oraciones, luego por espacios — respetando la coherencia semántica
    divisor = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # máximo 600 caracteres por fragmento
    chunk_overlap=200,      # 80 caracteres de solapamiento entre fragmentos
    separators=["\n\n", "\n", ". ", " "]
    )

    fragmentos = divisor.split_documents(documentos_crudos)
    print(f"✓ Fragmentación completada")
    print(f"  Documentos originales: {len(documentos_crudos)} páginas")
    print(f"  Fragmentos generados:  {len(fragmentos)}")

    # Mostramos un fragmento de ejemplo para verificar
    print(f"\n◈ Ejemplo de fragmento:")
    print(f"  Fuente: {fragmentos[0].metadata.get('source', 'desconocida')}")
    print(f"  Página: {fragmentos[0].metadata.get('page', '?')}")
    print(f"  Texto:  {fragmentos[0].page_content[:200]}...")
    return fragmentos

# --- Inicializar el modelo de Embeddings ---
def obtener_modelo_embeddings(nombre_modelo="intfloat/multilingual-e5-large"):
    # multilingual-e5-large corre localmente sin consumir API
    # Entiende bien el español rioplatense y múltiples idiomas
    modelo_embeddings = SentenceTransformerEmbeddings(
        model_name=nombre_modelo
        )
    print("✓ Modelo de embeddings configurado")
    print(f"  {nombre_modelo}, sin API")
    return modelo_embeddings


# --- Guardar en la Base Vectorial ---
def guardar_en_vector_db(fragmentos: list, embeddings):
    """Toma los fragmentos y los indexa en ChromaDB local."""
    print(f"\n💾 Guardando {len(fragmentos)} fragmentos en ChromaDB...")

    vectorstore = Chroma.from_documents(documents=fragmentos,
                                        embedding=embeddings,
                                        collection_name="proyecto_rag",
                                        persist_directory=DIRECTORIO_CHROMA
                                        )

    print(f"✓ Base vectorial lista")
    print(f"  Directorio: {DIRECTORIO_CHROMA}")
    print(f"  Fragmentos indexados: {len(fragmentos)}")
    print("✓ Base vectorial actualizada y guardada en disco.")
    return vectorstore



# --- PLa función que unifica todo el pipeline ---
def ejecutar_pipeline_ingesta():
    """Función orquestadora que corre todo el proceso de punta a punta."""
    print("=== INICIANDO PIPELINE DE INGESTA RAG ===")

    # 1. Cargar y partir
    fragmentos = cargar_y_fragmentar_pdfs(CARPETA_DATOS)
    if not fragmentos:
        print("❌ Proceso abortado: No hay fragmentos para indexar.")
        return

    # 2. Obtener embeddings
    embeddings = obtener_modelo_embeddings()

    # 3. Indexar en la base de datos
    vector_db = guardar_en_vector_db(fragmentos, embeddings)

    print("\n🚀 ¡Sistema RAG alimentado con éxito!")
    return vector_db