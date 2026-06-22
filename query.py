from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import MODEL_NAME, NUM_CTX


def formatear_documentos(docs):
    """Une los fragmentos recuperados en un solo bloque de texto."""
    return "\n\n".join(doc.page_content for doc in docs)


def inicializar_pipeline_rag(vectorstore):
    """Configura el LLM local, el prompt y ensambla la cadena (LCEL) del RAG."""
    print("\n🤖 Conectando con el servidor Ollama local...")

    # 1. Configurar el LLM local
    llm = OllamaLLM(
        model=MODEL_NAME,
        temperature=0.1,  # preciso y reproducible
        num_ctx=NUM_CTX,
    )

    # Verificación rápida de salud del servicio local
    try:
        respuesta_prueba = llm.invoke(
            "Respondé solo con 'ok' si estás funcionando."
        )
        print(f"  ✓ Ollama responde: {respuesta_prueba.strip()}")
        print(f"  Modelo activo: {MODEL_NAME}")
    except Exception as e:
        raise RuntimeError(
            f"No se pudo conectar con Ollama. ¿Está corriendo el servidor? Error: {e}"
        )

    # 2. Configurar el Retriever (trae los 3 fragmentos más relevantes)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. Definir el prompt del sistema
    TEMPLATE = """Sos un asistente experto en Arduino y sensores que responde preguntas basándose ÚNICAMENTE en los documentos proporcionados.
                Si la respuesta no está en los documentos, decilo claramente: no la inventes.
                Documentos:
                {context}
                Pregunta: {question}
                Respuesta:"""

    prompt = PromptTemplate(
        template=TEMPLATE, input_variables=["context", "question"]
    )

    # 4. Ensamblar la cadena usando LangChain Expression Language (LCEL)
    pipeline_rag = (
        {
            "context": retriever | formatear_documentos,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✓ Pipeline RAG configurado con éxito.")
    print("  Flujo: pregunta → ChromaDB (k=3) → prompt → Ollama → respuesta\n")

    return pipeline_rag


def hacer_pregunta(pipeline_rag, pregunta: str) -> str:
    """Ejecuta una consulta sobre el pipeline RAG ensamblado."""
    print(f"\n🔍 Buscando respuesta para: '{pregunta}'...")
    respuesta = pipeline_rag.invoke(pregunta)
    return respuesta