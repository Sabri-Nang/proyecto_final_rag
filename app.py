from pathlib import Path
import gradio as gr
from indexer import cargar_pdfs_interfaz
from query import pipeline_rag, retriever
import config

def responder_pregunta(pregunta, historial):
    if not pregunta.strip():
        return historial, ""
        
    respuesta = pipeline_rag.invoke(pregunta)
    fragmentos_fuente = retriever.invoke(pregunta)
    
    lineas_fuente = []
    for frag in fragmentos_fuente:
        fuente = Path(frag.metadata.get("source", "desconocida")).name
        pagina = frag.metadata.get("page", "?")
        lineas_fuente.append(f"• {fuente} (pág. {pagina})")
        
    historial = historial + [
        {"role": "user",      "content": pregunta},
        {"role": "assistant", "content": respuesta}
    ]
    return historial, "\n".join(lineas_fuente)

# ─── Interfaz Gradio ───
if config.MODO_LOCAL:
    title = "RAG Local - IFTS24"
else:
    title = "RAG con HuggingFace Spaces"
with gr.Blocks(title=title, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Asistente Inteligente de Documentación Técnica para Robótica")
    gr.Markdown("**Laboratorio de PLN — IFTS24, 2026**")
    gr.Markdown("Alumna: Sabrina Sanches")

    with gr.Tab("📄 Cargar documentos"):
        upload_component = gr.File(
            label="Seleccioná tus PDFs",
            file_types=[".pdf"],
            file_count="multiple"
        )
        boton_cargar = gr.Button("Indexar documentos", variant="primary")
        estado_carga = gr.Textbox(label="Estado", interactive=False, lines=3)
        
        boton_cargar.click(
            fn=cargar_pdfs_interfaz,
            inputs=[upload_component],
            outputs=[estado_carga]
        )

    with gr.Tab("💬 Hacer preguntas"):
        chatbot_componente = gr.Chatbot(label="Conversación", height=400)
        with gr.Row():
            pregunta_componente = gr.Textbox(
                label="Tu pregunta",
                placeholder="¿Qué dice el documento sobre...?",
                scale=4
            )
            boton_preguntar = gr.Button("Preguntar", variant="primary", scale=1)
            
        fuentes_componente = gr.Textbox(
            label="Fragmentos consultados",
            interactive=False,
            lines=3
        )
        
        # Conexión de eventos
        boton_preguntar.click(
            fn=responder_pregunta,
            inputs=[pregunta_componente, chatbot_componente],
            outputs=[chatbot_componente, fuentes_componente]
        )
        pregunta_componente.submit(
            fn=responder_pregunta,
            inputs=[pregunta_componente, chatbot_componente],
            outputs=[chatbot_componente, fuentes_componente]
        )

if __name__ == "__main__":
    demo.launch()