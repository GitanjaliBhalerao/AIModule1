# main.py
import os
import logging
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_azure_ai.embeddings import AzureAIOpenAIApiEmbeddingsModel
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# ─── Logging Setup ───────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag_app")

# ─── Base Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Helper Functions to load pdf file ────────────────────────────────────────────────────────
def load_and_split_pdf(pdf_path, chunk_size=1000, chunk_overlap=200):
    logger.info(f"Loading PDF: {pdf_path}")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    docs = splitter.split_documents(pages)
    logger.info(f"Split into {len(docs)} chunks")
    return docs

# ─── Helper Functions build FAISS index ────────────────────────────────────────────────────────
def build_faiss_index(docs, embed_model, index_path="faiss_index"):
    full_index_path = os.path.join(BASE_DIR, index_path)

    if os.path.isdir(full_index_path):
        import shutil
        shutil.rmtree(full_index_path)
        logger.info("Removed existing FAISS index")

    vs = FAISS.from_documents(docs, embed_model)
    vs.save_local(full_index_path)
    logger.info(f"FAISS index built and saved to {full_index_path}")

    return vs
# ─── Helper Functions for RAG chain ────────────────────────────────────────────────────────
# FAISS vector store is converted into a retriever
#here k=2 means retriever will fetch the top 2 most relavant chunks for a query
def create_rag_chain(vs, chat_model, k=2):
    logger.info(f"Creating RAG pipeline with k={k}")

    retriever = vs.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant.
        Use ONLY the following context to answer the question.
        If the answer is not found, say "I don't know".

        Context:
        {context}

        Question:
        {input}
        """
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

# ─── Core Langchain Pipeline , takes user query , retrieve relavant docs and format them 
# passes them into the prompt to chat model and finally parese the output as string ────────────────────────────────────────────────────────
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "input": RunnablePassthrough(),
        }
        | prompt
        | chat_model
        | StrOutputParser()
    )

    logger.info("✅ RAG chain ready")
    return rag_chain, retriever


# ─── App Initialization ─────────────────────────────────────────────────────
load_dotenv()

# Direct OpenAI-compatible endpoint + API key
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = os.getenv("AZURE_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("AZURE_CHAT_MODEL", "gpt-4o")
PDF_PATH = os.getenv("PDF_FILE_PATH", "Techvarsity.pdf")
TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.3"))

# Convert relative PDF path to absolute path
if not os.path.isabs(PDF_PATH):
    PDF_PATH = os.path.join(BASE_DIR, PDF_PATH)

INDEX_HTML_PATH = os.path.join(BASE_DIR, "index.html")

# Debug logs
logger.info(f"OPENAI_BASE_URL loaded: {'Yes' if OPENAI_BASE_URL else 'No'}")
logger.info(f"OPENAI_API_KEY loaded: {'Yes' if OPENAI_API_KEY else 'No'}")
logger.info(f"CHAT_MODEL: {CHAT_MODEL}")
logger.info(f"EMBED_MODEL: {EMBED_MODEL}")
logger.info(f"PDF_PATH: {PDF_PATH}")

if not OPENAI_BASE_URL:
    raise ValueError("Missing OPENAI_BASE_URL in environment variables")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment variables")

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF file not found: {PDF_PATH}")

if not os.path.exists(INDEX_HTML_PATH):
    raise FileNotFoundError(f"index.html not found: {INDEX_HTML_PATH}")

# Make sure the LangChain Azure AI client can pick them up
os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

logger.info("Initializing Azure AI / Azure OpenAI clients")

# Embeddings client
embeddings = AzureAIOpenAIApiEmbeddingsModel(
    model=EMBED_MODEL,
)

# Chat client
chat_model = AzureAIOpenAIApiChatModel(
    model=CHAT_MODEL,
    temperature=TEMPERATURE,
)

# Build index & RAG chain
docs = load_and_split_pdf(PDF_PATH)
vs = build_faiss_index(docs, embeddings)
rag_chain, retriever = create_rag_chain(vs, chat_model)

# ─── Flask App ───────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True) or {}
        query = data.get("query", "").strip()

        logger.info(f"Query received: {query!r}")

        # Empty input / greeting fallback
        if query.lower() in {"", "hi", "hello", "hey"}:
            resp = chat_model.invoke(query or "Hello!")
            return jsonify(answer=getattr(resp, "content", str(resp)), sources=[])

        # Retrieve documents manually (modern LangChain usage)
        docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} documents")

        # If no docs → fallback to direct model answer
        if not docs:
            resp = chat_model.invoke(query)
            return jsonify(answer=getattr(resp, "content", str(resp)), sources=[])

        # Run RAG chain
        answer = rag_chain.invoke(query)

        # Extract sources
        sources = [
            f"page {d.metadata.get('page', '?')}: {d.page_content[:100].replace(chr(10), ' ')}…"
            for d in docs
        ]

        return jsonify(answer=str(answer), sources=sources)

    except Exception as e:
        logger.exception("Error in /api/chat")
        # During development, return real error
        return jsonify(error=str(e)), 500


@app.route("/")
def index():
    return send_file(INDEX_HTML_PATH)


if __name__ == "__main__":
    logger.info("Starting Flask on http://0.0.0.0:8000")
    app.run(host="0.0.0.0", port=8000, debug=True)
