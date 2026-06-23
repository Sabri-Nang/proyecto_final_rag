from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from indexer2 import vectorstore
import config

# Configuración directa con el nuevo modelo configurado en tu config.py
llm = HuggingFaceEndpoint(
    repo_id=config.MODEL_ID,
    huggingfacehub_api_token=config.HF_TOKEN,
    temperature=0.1,
    max_new_tokens=512
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def formatear_documentos(docs):
    return "\n\n".join(doc.page_content for doc in docs)

TEMPLATE = """Sos un asistente experto en Arduino y sensores que responde preguntas basándose ÚNICAMENTE en los documentos proporcionados.
                Si la respuesta no está en los documentos, decilo claramente: no la inventes.
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
