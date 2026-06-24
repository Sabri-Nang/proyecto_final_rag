from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from indexer import vectorstore
import config

# ─── Inicialización del LLM Dinámico ───
if config.MODO_LOCAL:
    print(f"🤖 Inicializando LLM Local vía Ollama: {config.MODEL_NAME}...")
    llm = Ollama(
        model=config.MODEL_NAME, 
        temperature=0.1,
        num_ctx=config.NUM_CTX
    )
else:
    print(f"☁️ Inicializando LLM Cloud vía Hugging Face Endpoint: {config.MODEL_ID}...")
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=config.MODEL_ID,
        task="conversational",
        huggingfacehub_api_token=config.HF_TOKEN,
        temperature=0.1,
        max_new_tokens=512
    )
    llm = ChatHuggingFace(llm=llm_endpoint)
    

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def formatear_documentos(docs):
    return "\n\n".join(doc.page_content for doc in docs)

TEMPLATE = """Sos un tutor de robótica y electrónica embebida (experto en Arduino, microcontroladores, 
                sensores y IoT). Tu objetivo es ayudar a estudiantes y makers a comprender el hardware basándote 
                ÚNICAMENTE en los documentos proporcionados.
                Si la respuesta no está en los documentos, decilo claramente: no la inventes.
                Reglas estrictas de comportamiento:
                1. Si la respuesta o el código de ejemplo no se encuentran explícitamente en los 
                documentos de contexto, respondé amablemente: "Lo siento, esa información técnica no 
                está disponible en el manual provisto". No inventes pines, voltajes ni librerías bajo 
                ninguna circunstancia.
                2. Cuando incluyas código de programación (C++ de Arduino, Python, etc.), usá 
                OBLIGATORIAMENTE bloques de código con formato Markdown (por ejemplo, ```cpp ... ```) 
                y agregá comentarios breves en las líneas principales para explicar su funcionamiento.
                3. Destaca siempre los datos críticos de hardware en **negrita** 
                (ej. voltajes de operación como **3.3V** o **5V**, pines de conexión como **VCC**, 
                **GND**, **DATA** y rangos de medición).
                4. Si la documentación contiene tablas de especificaciones o diagramas de pines 
                descritos en texto, organizá tu respuesta usando tablas de Markdown para que sea 
                altamente legible y fácil de escanear a simple vista.
                Documentos:
                {context}
                Pregunta: {question}
                Respuesta:"""

prompt = PromptTemplate(
    template=TEMPLATE,
    input_variables=["context", "question"]
)

pipeline_rag = (
    {"context": retriever | formatear_documentos, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)