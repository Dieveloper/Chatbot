import os
import unicodedata
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from fastapi.middleware.cors import CORSMiddleware

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "portfolio-diego-493515-fcba40f7ae88.json"
# --- CONFIGURACIÓN DE ENTORNO ---
# Asegúrate de que tu archivo JSON de credenciales esté en la carpeta del proyecto
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "tu-archivo-credenciales.json"




# --- UTILIDADES ---
def normalizar_texto(texto):
    """Limpia el texto de tildes, signos y mayúsculas para ahorrar tokens."""
    texto = texto.lower().strip()
    # Eliminamos signos comunes
    for caracter in ["?", "!", "¿", "¡", ".", ","]:
        texto = texto.replace(caracter, "")
    # Eliminamos acentos
    texto = unicodedata.normalize('NFD', texto)
    return ''.join(c for c in texto if unicodedata.category(c) != 'Mn')

app = FastAPI(
    title="Diego's Gemini RAG API",
    description="Backend profesional con FastAPI y Google Gemini para consulta de CV"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Esto permite que cualquier web consulte tu API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- 1. PROCESAMIENTO DEL CV (ETL) ---
# Cargamos el PDF una sola vez al arrancar la API
PDF_PATH = "data/model.pdf"

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"No se encontró el archivo en {PDF_PATH}. Crea la carpeta 'data' y sube tu PDF.")

loader = PyPDFLoader(PDF_PATH)
documentos = loader.load()

# Dividimos el texto en trozos (chunks) para que Gemini los procese mejor
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documentos)

# --- 2. VECTOR STORE (CEREBRO) ---
# Usamos los Embeddings oficiales de Google
# Usamos el modelo más nuevo y estándar de Google
embeddings_google = VertexAIEmbeddings(
    model_name="text-embedding-004",
    location="europe-west1"
)
# Creamos la base de datos vectorial en memoria (o persistente)
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings_google,
    persist_directory="./db_portfolio_google"
)

# --- 3. MODELO GEMINI Y PROMPT ---
llm = VertexAI(
    model_name="gemini-2.5-flash-lite", 
    location="europe-west1", 
    temperature=0
)
system_prompt = (
    "Eres un asistente experto en analizar currículums. "
    "Tu objetivo es ayudar a los reclutadores a conocer mejor a Diego Gomez Jordan. "
    "Responde de forma profesional, concisa y amable basándote solo en este contexto: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Creamos la cadena de recuperación (RAG Chain)
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vector_db.as_retriever(), combine_docs_chain)

# --- MODELOS DE DATOS ---
class Query(BaseModel):
    question: str

class Response(BaseModel):
    user_question: str
    ai_answer: str

# --- ENDPOINTS ---
@app.get("/")
def home():
    return {
        "message": "API de Diego Gomez Jordan conectada a Google Gemini",
        "docs": "/docs",
        "status": "Ready"
    }


SALUDOS = {
    "hola", "hola!", "buenas", "buenos dias", "buenas tardes", "buenas noches",
    "hey", "hola que tal", "saludos", "hola diego ai", "hola ai", "que tal",
    "buen dia", "hola!", "holaaa", "hi", "hello"
}


DESPEDIDAS = {
    "gracias", "muchas gracias", "perfecto gracias", "adios", "chao", 
    "hasta luego", "nos vemos", "bye", "ok gracias", "entendido", 
    "genial", "gracias por la info", "mil gracias", "listo"
}

@app.post("/ask", response_model=Response)
async def ask_cv(query: Query):
    """
    Endpoint principal para preguntar sobre el CV de Diego.
    """
    # 1. Obtenemos el texto y lo normalizamos
    # Usamos query.question porque FastAPI ya parseó el JSON por nosotros
    texto_usuario = query.question
    query_normalizada = normalizar_texto(texto_usuario)

    # 2. Filtro de Saludos (Ahorro de tokens)
    if query_normalizada in SALUDOS:
        return {
            "user_question": texto_usuario,
            "ai_answer": "¡Hola! 👋 Soy el asistente virtual de Diego. ¿En qué puedo ayudarte hoy respecto a su perfil profesional?"
        }

    # 3. Filtro de Despedidas
    elif query_normalizada in DESPEDIDAS:
        return {
            "user_question": texto_usuario,
            "ai_answer": "¡De nada! Si tienes más dudas sobre los proyectos de Diego, aquí estaré. ¡Un saludo!"
        }

    try:
            # 4. Ejecutamos la cadena RAG solo si no es saludo/despedida
            # Usamos el texto original (texto_usuario) para no perder semántica en la IA
        result = rag_chain.invoke({"input": texto_usuario})
            
        return {
                "user_question": texto_usuario,
                "ai_answer": result['answer']
        }
    except Exception as e:
            # Es mejor imprimir el error en consola para debuguear
        print(f"Error detectado: {e}")
        raise HTTPException(status_code=500, detail="Error en la inferencia del nodo")

# --- EJECUCIÓN ---
# Para arrancar: uvicorn main:app --reload