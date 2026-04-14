from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# --- 1. CARGA Y PROCESAMIENTO ---
loader = PyPDFLoader("data/model.pdf")
documentos = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documentos)

# --- 2. VECTOR STORE ---
embeddings_locales = OllamaEmbeddings(model="llama3")
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings_locales,
    persist_directory="./mi_db_vectorial"
)

# --- 3. MODELO Y PROMPT ---
llm = OllamaLLM(model="llama3")

system_prompt = (
    "Eres un asistente experto en analizar currículums. "
    "Responde solo basándote en el contexto proporcionado: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# --- 4. LA CADENA ---
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vector_db.as_retriever(), combine_docs_chain)

# --- 5. EJECUCIÓN ---
while True:
    pregunta = input("\nPregunta: ")
    
    if pregunta.upper() != "NADA":
        respuesta = rag_chain.invoke({"input": pregunta})
        print(f"Respuesta: {respuesta['answer']}")
    else:
        break