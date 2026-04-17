# 🧠 Diego's Neural Lab: CV Chatbot con RAG, FastAPI & Google Gemini

Este proyecto es un asistente inteligente de alto rendimiento diseñado para interactuar con mi perfil profesional. Utiliza una arquitectura **RAG** (Generación Aumentada por Recuperación) para ofrecer respuestas precisas, veraces y contextualizadas basadas en mi trayectoria técnica.

Actualmente, el sistema funciona como una **API REST independiente**, permitiendo su integración en cualquier frontend (como mi portfolio personal).

## 🚀 Stack Tecnológico
* **Google Gemini 1.5 Flash / 2.5 Flash Lite**: Modelos de lenguaje de última generación vía **Vertex AI**.
* **FastAPI**: Framework de alto rendimiento para la construcción de la API.
* **LangChain**: Orquestador principal para la lógica de recuperación y cadenas de IA.
* **ChromaDB**: Base de datos vectorial para el almacenamiento y búsqueda de embeddings.
* **Google Cloud (Vertex AI)**: Infraestructura cloud para inferencia y generación de embeddings (`text-embedding-004`).
* **Python 3.12+**: Lenguaje base del ecosistema.

## 🏗️ Arquitectura del Sistema
1.  **Ingesta de Datos (ETL)**: Procesamiento de mi CV en PDF mediante `PyPDFLoader`.
2.  **Segmentación (Chunking)**: División estratégica del texto para optimizar la ventana de contexto del modelo.
3.  **Vectorización**: Generación de embeddings vectoriales mediante modelos oficiales de Google Cloud.
4.  **Almacenamiento**: Persistencia en **ChromaDB** para una recuperación semántica ultra rápida.
5.  **Inferencia RAG**: Cadena de recuperación que combina el contexto del CV con la potencia de Gemini para responder de forma profesional.

## 🛠️ Instalación y Configuración

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/Dieveloper/Chatbot.git
   cd chat