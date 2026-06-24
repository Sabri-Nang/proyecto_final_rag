# 🔌 Asistente RAG de Documentación Técnica para Robótica

Sistema de **Retrieval-Augmented Generation (RAG)** que responde preguntas sobre
manuales y datasheets de Arduino y sensores, basándose únicamente en el
contenido de los PDFs cargados — sin inventar información que no esté en los
documentos.

> Trabajo final — Procesamiento del Lenguaje Natural, IFTS24 (2026)
> Alumna: Sabrina Sanches

---

## ¿Qué hace?

1. Recibe **PDFs** (manuales de sensores, datasheets, guías de Arduino) desde
   la propia interfaz y utiliza los precargados en la carpeta `documentos/`.
2. La app los **fragmenta** los documentos, genera **embeddings** y los guarda en una base
   vectorial **ChromaDB**.
3. Al hacer una pregunta, busca los fragmentos más relevantes y se los pasa a
   un **LLM**, que redacta la respuesta citando de qué documento y página salió
   cada fragmento usado.

```
PDF ─▶ Fragmentación ─▶ Embeddings ─▶ ChromaDB (memoria)
                                          │
Pregunta ──────────────▶ Búsqueda (top-3) ┘
                              │
                              ▼
                     Prompt + contexto ─▶ LLM ─▶ Respuesta
```

## Tecnologías

| Componente         | Herramienta                                          |
|---------------------|-------------------------------------------------------|
| Orquestación RAG    | [LangChain](https://www.langchain.com/)                |
| Embeddings          | `intfloat/multilingual-e5-large` (Sentence Transformers) |
| Base vectorial      | [ChromaDB](https://www.trychroma.com/) (en memoria)     |
| LLM (local)         | [Ollama](https://ollama.com/) — `gemma4:e2b`             |
| LLM (nube)          | Hugging Face Inference Endpoint — `Qwen2.5-Coder-7B-Instruct` |
| Interfaz            | [Gradio](https://www.gradio.app/)                       |

## Estructura del proyecto

```
.
├── app.py              # Interfaz Gradio (pestañas: cargar, preguntar, acerca de)
├── config.py            # Configuración centralizada (modelo, paths, chunking)
├── indexer.py            # Carga de PDFs, fragmentación e indexado en ChromaDB
├── query.py              # Inicialización del LLM y la cadena RAG
├── requirements.txt       # Dependencias
├── documentos/            # PDFs que se indexan automáticamente al arrancar
└── .gitignore
```


## Instalación y uso local

Requiere Python 3.10+ y [Ollama](https://ollama.com/) corriendo en tu máquina,
con el modelo configurado en `config.py` (`MODEL_NAME`) ya descargado, por
ejemplo:

```bash
ollama pull gemma4:e2b   # o el modelo que tengas seteado en config.py
```

```bash
git clone <url-de-este-repo>
cd <carpeta-del-repo>

python -m venv .venv
source .venv/bin/activate      # En Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Colocá tus PDFs en la carpeta `documentos/` y corré:

```bash
python app.py
```

La app va a estar disponible en `http://localhost:7860`.

> `config.py` detecta el entorno automáticamente: si no encuentra la variable
> de entorno `SPACE_ID` (la define Hugging Face Spaces), asume que estás en
> modo local y usa Ollama. No es necesario tocar nada a mano.

## Despliegue en Hugging Face Spaces

1. Creá un nuevo Space, tipo **Gradio**.
2. Subí todos los archivos del repo **incluyendo la carpeta `documentos/`**.
3. En **Settings → Variables and secrets**, agregá un secreto:

   | Nombre     | Valor                                              |
   |------------|-----------------------------------------------------|
   | `HF_TOKEN` | Tu token de Hugging Face (con permiso de lectura como mínimo) |

   Lo generás en [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
4. El Space va a detectar automáticamente que está en la nube (gracias a la
   variable `SPACE_ID` que define Hugging Face) y va a usar el endpoint de
   `Qwen2.5-Coder-7B-Instruct` en vez de Ollama.


## Agregar más documentos después de desplegado

Desde la pestaña **📄 Cargar documentos** de la interfaz podés subir PDFs
adicionales en cualquier momento; se suman a los que ya estén indexados. Esos
PDFs adicionales **no persisten** si el Space se reinicia — para que un
documento esté siempre disponible, sumalo a la carpeta `documentos/` del
repositorio.

## Limitaciones conocidas

- La base vectorial es en memoria: se pierde al reiniciar el Space (se
  reconstruye sola desde `documentos/`, pero los PDFs subidos a mano desde la
  interfaz no quedan guardados de forma permanente).
- El LLM responde solo en base a los fragmentos recuperados; si la
  información no está en los PDFs cargados, lo va a indicar explícitamente en
  vez de inventar una respuesta.
- Pensado para corpus chicos/medianos (decenas de PDFs); para volúmenes
  grandes conviene una base vectorial persistente y, eventualmente, un
  servicio de embeddings dedicado.

## Créditos

Proyecto final Procesamiento del Lenguaje Natural —
IFTS24, 2026. Alumna: Sabrina Sanches.
