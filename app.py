from pathlib import Path
import gradio as gr
from indexer import cargar_pdfs_interfaz, FRAGMENTOS_INICIALES, ARCHIVOS_INICIALES
from query import pipeline_rag, retriever
import config


def responder_pregunta(pregunta, historial):
    if not pregunta.strip():
        return historial, ""

    respuesta = pipeline_rag.invoke(pregunta)
    fragmentos_fuente = retriever.invoke(pregunta)

    bloques_fuente = []
    for i, frag in enumerate(fragmentos_fuente, start=1):
        fuente = Path(frag.metadata.get("source", "desconocida")).name
        pagina = frag.metadata.get("page", "?")
        texto = frag.page_content.strip()
        bloques_fuente.append(
            f"📎 Fragmento {i} — {fuente} · pág. {pagina}\n{texto}"
        )

    historial = historial + [
        {"role": "user", "content": pregunta},
        {"role": "assistant", "content": respuesta}
    ]
    return historial, "\n\n────────────────────\n\n".join(bloques_fuente)


def limpiar_conversacion():
    return [], "", ""


# ─── Estado de la base de conocimiento al arrancar ───
if ARCHIVOS_INICIALES:
    lista_archivos = "\n".join(f"  • {nombre}" for nombre in ARCHIVOS_INICIALES)
    ESTADO_INICIAL = (
        f"✓ Base de conocimiento lista: {len(ARCHIVOS_INICIALES)} documento(s) "
        f"precargado(s), {FRAGMENTOS_INICIALES} fragmentos indexados.\n{lista_archivos}"
    )
else:
    ESTADO_INICIAL = (
        "⚠ No hay documentos precargados todavía. Subí uno o más PDFs en esta "
        "pestaña, o agregalos a la carpeta `documentos/` del proyecto antes de "
        "desplegar para que se indexen solos al arrancar."
    )

EJEMPLOS_PREGUNTAS = [
    "¿Qué rango de humedad puede medir el sensor?",
    "¿Cómo se conecta el sensor a una placa Arduino?",
    "¿Qué precisión tiene la medición de temperatura?",
]

# ─── Identidad visual ───
# Paleta: papel cálido + verde placa de circuito + cobre, con un guiño a la
# fila de pines de un microcontrolador como elemento distintivo del header.
TEMA = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Space Grotesk"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#FAFAF7",
    body_background_fill_dark="#11161B",
    block_background_fill="#FFFFFF",
    block_border_color="#E3E0D6",
    block_title_text_weight="600",
    button_primary_background_fill="#0E8C8C",
    button_primary_background_fill_hover="#0B6F6F",
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="#F3EEE4",
    button_secondary_text_color="#1B2430",
)

CSS_PERSONALIZADO = """
.cabecera-rag {
    padding: 22px 26px 18px 26px;
    border-radius: 14px;
    background: linear-gradient(135deg, #0E8C8C 0%, #0B6F6F 100%);
    color: #FAFAF7;
    margin-bottom: 6px;
}
.cabecera-rag h1 {
    margin: 0 0 4px 0;
    font-size: 1.55rem;
    letter-spacing: -0.01em;
}
.cabecera-rag p {
    margin: 2px 0;
    opacity: 0.92;
    font-size: 0.92rem;
}
.fila-pines {
    display: flex;
    gap: 5px;
    margin-top: 14px;
}
.fila-pines span {
    width: 9px;
    height: 9px;
    border-radius: 2px;
    background: #FAFAF7;
    opacity: 0.55;
}
.fila-pines span:nth-child(3n+2) { background: #E8A560; opacity: 0.95; }
#fuentes-box textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}
#estado-box textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}
"""

HEADER_HTML = """
<div class="cabecera-rag">
  <h1>🔌 Asistente Inteligente de Documentación Técnica para Robótica</h1>
  <p>Laboratorio de Procesamiento de Lenguaje Natural — IFTS24, 2026</p>
  <p>Alumna: Sabrina Sanches</p>
  <div class="fila-pines">
    <span></span><span></span><span></span><span></span><span></span>
    <span></span><span></span><span></span><span></span><span></span>
    <span></span><span></span><span></span><span></span><span></span>
  </div>
</div>
"""

ACERCA_DE_MD = f"""
### ¿Qué hace esta app?

Es un sistema **RAG (Retrieval-Augmented Generation)**: en vez de que el modelo
de lenguaje invente respuestas, primero busca los fragmentos más relevantes
dentro de tus PDFs (manuales de sensores, datasheets, guías de Arduino) y
recién después redacta la respuesta basándose únicamente en esos fragmentos.

**Pipeline:**
1. 📄 Los PDFs se dividen en fragmentos de ~{config.CHUNK_SIZE} caracteres.
2. 🧮 Cada fragmento se convierte en un vector con `{config.EMBEDDING_MODEL}`.
3. 🗄️ Los vectores se guardan en una base **ChromaDB** en memoria.
4. 🔎 Al preguntar, se buscan los 3 fragmentos más parecidos a la pregunta.
5. 🤖 Un LLM redacta la respuesta usando solo esos fragmentos como contexto.

**Modelo de lenguaje activo:** {"Ollama · " + config.MODEL_NAME if config.MODO_LOCAL else "Hugging Face Endpoint · " + config.MODEL_ID}

**Importante:** la base vectorial vive en memoria. Si el Space se reinicia o
se "duerme", se vuelve a construir sola a partir de los PDFs de la carpeta
`documentos/` — no es necesario (ni conviene) subir una carpeta `chroma_db`.
"""

if config.MODO_LOCAL:
    titulo = "RAG Local - IFTS24"
else:
    titulo = "RAG con HuggingFace Spaces"

with gr.Blocks(title=titulo) as demo:
    gr.HTML(HEADER_HTML)

    with gr.Tab("📄 Cargar documentos"):
        estado_inicial_componente = gr.Textbox(
            value=ESTADO_INICIAL,
            label="Base de conocimiento actual",
            interactive=False,
            lines=4,
            elem_id="estado-box"
        )
        gr.Markdown("Sumá más PDFs a la conversación (se agregan a los ya indexados):")
        upload_component = gr.File(
            label="Seleccioná tus PDFs",
            file_types=[".pdf"],
            file_count="multiple"
        )
        boton_cargar = gr.Button("➕ Indexar documentos", variant="primary")
        estado_carga = gr.Textbox(label="Resultado de la indexación", interactive=False, lines=3)

        boton_cargar.click(
            fn=cargar_pdfs_interfaz,
            inputs=[upload_component],
            outputs=[estado_carga]
        )

    with gr.Tab("💬 Hacer preguntas"):
        chatbot_componente = gr.Chatbot(
            label="Conversación",
            height=420,
        )
        with gr.Row():
            pregunta_componente = gr.Textbox(
                label="Tu pregunta",
                placeholder="¿Qué dice el documento sobre...?",
                scale=4
            )
            boton_preguntar = gr.Button("Preguntar", variant="primary", scale=1)

        with gr.Row():
            boton_limpiar = gr.Button("🧹 Nueva conversación", variant="secondary", size="sm")

        fuentes_componente = gr.Textbox(
            label="Fragmentos consultados",
            interactive=False,
            lines=10,
            max_lines=40,
            elem_id="fuentes-box"
        )

        gr.Examples(
            examples=EJEMPLOS_PREGUNTAS,
            inputs=[pregunta_componente],
            label="Preguntas de ejemplo"
        )

        # Conexión de eventos
        boton_preguntar.click(
            fn=responder_pregunta,
            inputs=[pregunta_componente, chatbot_componente],
            outputs=[chatbot_componente, fuentes_componente]
        ).then(lambda: "", outputs=[pregunta_componente])

        pregunta_componente.submit(
            fn=responder_pregunta,
            inputs=[pregunta_componente, chatbot_componente],
            outputs=[chatbot_componente, fuentes_componente]
        ).then(lambda: "", outputs=[pregunta_componente])

        boton_limpiar.click(
            fn=limpiar_conversacion,
            outputs=[chatbot_componente, fuentes_componente, pregunta_componente]
        )

    with gr.Tab("ℹ️ Acerca de"):
        gr.Markdown(ACERCA_DE_MD)

if __name__ == "__main__":
    demo.launch(theme=TEMA, css=CSS_PERSONALIZADO)