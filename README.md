# 📄 CV Chatbot con RAG & Ollama

Este es un asistente inteligente capaz de responder preguntas sobre un Currículum Vitae (PDF) utilizando una arquitectura **RAG** (Generación Aumentada por Recuperación). El sistema procesa el documento localmente, asegurando la privacidad de los datos.

## 🚀 Tecnologías utilizadas
* **LangChain**: Framework para orquestar la lógica de LLMs.
* **Ollama (Llama 3)**: Modelo de lenguaje ejecutado de forma local.
* **ChromaDB**: Base de datos vectorial para el almacenamiento de embeddings.
* **Python 3.14+**: Lenguaje de programación base.

## 🛠️ Funcionamiento
1. **Carga**: Se lee el archivo PDF mediante `PyPDFLoader`.
2. **Fragmentación**: El texto se divide en trozos (chunks) para un mejor procesamiento.
3. **Embeddings**: Se convierten los textos a vectores usando `OllamaEmbeddings`.
4. **Recuperación**: El sistema busca los fragmentos más relevantes ante una pregunta.
5. **Respuesta**: El LLM genera una respuesta basada exclusivamente en el contexto del CV.

## 📋 Requisitos e Instalación
1. Tener instalado [Ollama](https://ollama.com/) y el modelo Llama3 (`ollama run llama3`).
2. Instalar las dependencias de Python:
   ```bash
   pip install langchain langchain-community langchain-ollama langchain-chroma pypdf