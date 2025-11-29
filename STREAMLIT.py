# STREAMLIT.py — Final (Chroma + retry + gpt-4o-mini-1.5)
import streamlit as st
import tempfile
import traceback
import time
import random
import json
from typing import Optional, List

# ------------------------------
# Document loaders
# ------------------------------
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

def load_document_from_streamlit(uploaded_file):
    file_name = uploaded_file.name.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_name) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    if file_name.endswith(".pdf"):
        loader = PyPDFLoader(temp_path)
    elif file_name.endswith(".docx"):
        loader = Docx2txtLoader(temp_path)
    elif file_name.endswith(".txt"):
        loader = TextLoader(temp_path)
    else:
        raise ValueError("Unsupported file format.")

    return loader.load()


# ------------------------------
# Text splitter
# ------------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    return splitter.split_documents(documents)


# ------------------------------
# Embeddings (SentenceTransformers)
# ------------------------------
from sentence_transformers import SentenceTransformer

@st.cache_resource
def get_embeddings_model():
    # small and stable model; will download on first run
    return SentenceTransformer("all-MiniLM-L6-v2")


# ------------------------------
# Chroma vector store (local)
# ------------------------------
from langchain_community.vectorstores import Chroma

def create_vector_store(chunks):
    model = get_embeddings_model()

    class Embedder:
        # texts: List[str] -> List[List[float]]
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            vectors = model.encode(texts)
            # model.encode returns numpy array; convert to lists
            return [v.tolist() for v in vectors]

        # text: str -> List[float]
        def embed_query(self, text: str) -> List[float]:
            vec = model.encode(text)
            # If encode returns 1D vector
            return vec.tolist()

    embeddings = Embedder()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_docs"
    )
    return vector_store


# ------------------------------
# Retriever
# ------------------------------
def get_retriever(vector_store, k: int = 5):
    return vector_store.as_retriever(search_kwargs={"k": k})


# ------------------------------
# Raw HTTP OpenAI LLM with retry/backoff (LangChain-compatible)
# ------------------------------
from langchain.llms.base import LLM

class RawOpenAIChatLLM(LLM):
    # Declare fields so LangChain (pydantic wrapped LLM) accepts them
    api_key: str
    model: str = "gpt-4o-mini-1.5"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini-1.5", **kwargs):
        # Use parent initializer to set pydantic fields correctly
        super().__init__(api_key=api_key, model=model, **kwargs)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        import requests

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        max_attempts = 6
        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
            except Exception as e:
                # network error — retry with backoff
                if attempt == max_attempts:
                    return f"NETWORK ERROR: {e}"
                backoff = min(20, 2 ** attempt + random.random())
                time.sleep(backoff)
                continue

            # parse json safely
            try:
                data = resp.json()
            except Exception as e:
                # invalid json
                if attempt == max_attempts:
                    return f"Invalid JSON response: {e}\nStatus code: {resp.status_code}\nBody: {resp.text}"
                time.sleep(1 + random.random())
                continue

            # Successful chat completion shape
            if resp.status_code == 200 and "choices" in data:
                try:
                    return data["choices"][0]["message"]["content"]
                except Exception:
                    # fallback stringification
                    return json.dumps(data, indent=2)

            # Rate limit / 429 handling
            # OpenAI sometimes returns 429 or a JSON error with code "rate_limit_exceeded"
            error = data.get("error") if isinstance(data, dict) else None
            code = None
            if isinstance(error, dict):
                code = error.get("code") or error.get("type") or error.get("message")

            if resp.status_code == 429 or (isinstance(code, str) and "rate_limit" in code):
                if attempt == max_attempts:
                    # final attempt failed
                    return json.dumps(data, indent=2)
                # exponential backoff with jitter
                wait = min(20, 2 ** attempt) + random.uniform(0, 1.5)
                time.sleep(wait)
                continue

            # Other 5xx server errors: retry
            if 500 <= resp.status_code < 600:
                if attempt == max_attempts:
                    return json.dumps(data, indent=2)
                time.sleep(min(10, 2 ** attempt))
                continue

            # If non-rate-limit client error or other error — return the error JSON for debugging
            return json.dumps(data, indent=2)

        # if we get here, return fallback
        return "❌ Unknown error: exceeded retries."

    @property
    def _llm_type(self):
        return "raw_openai_chat"

    @property
    def _identifying_params(self):
        return {"model": self.model}


# ------------------------------
# Helper: extract text from runnable output
# ------------------------------
def extract_text(obj) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    # langchain message object may have .content
    try:
        content = getattr(obj, "content", None)
        if isinstance(content, str):
            return content
    except Exception:
        pass
    # if dict like
    if isinstance(obj, dict):
        for key in ("content", "text", "output"):
            if key in obj and isinstance(obj[key], str):
                return obj[key]
    # fallback
    try:
        return str(obj)
    except Exception:
        return ""


# ------------------------------
# RAG chain (Runnable)
# ------------------------------
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

def build_rag_chain(retriever, model_name: str = "gpt-4o-mini-1.5"):
    llm = RawOpenAIChatLLM(api_key=st.secrets["OPENAI_API_KEY"], model=model_name)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use ONLY the context to answer the question. If no context exists, summarize the entire document.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    )

    chain = (
        RunnableMap({
            "context": lambda x: retriever.get_relevant_documents(x["question"]),
            "question": RunnablePassthrough(),
        })
        |
        (lambda x: {
            "context": "\n\n".join(doc.page_content for doc in x["context"]) if x["context"] else "NO_CONTEXT",
            "question": x["question"],
        })
        |
        prompt
        |
        llm
    )
    return chain


# ------------------------------
# Profession conclusion chain
# ------------------------------
def build_profession_chain(model_name: str = "gpt-4o-mini-1.5"):
    llm = RawOpenAIChatLLM(api_key=st.secrets["OPENAI_API_KEY"], model=model_name)

    prompt = PromptTemplate(
        input_variables=["profession", "rag_answer"],
        template="""
You are an expert {profession}. Write a short conclusion from that perspective using ONLY the document answer below.

DOCUMENT ANSWER:
{rag_answer}

CONCLUSION ({profession}):
"""
    )
    return prompt | llm


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="RAG Document Analyzer (Chroma + Retries)", layout="wide")
st.title("📄 RAG Document Analyzer — Chroma + Retry (gpt-4o-mini-1.5)")

uploaded_file = st.file_uploader("Upload your document (pdf/docx/txt)", type=["pdf", "docx", "txt"])
model_choice = "gpt-4o-mini-1.5"  # chosen default (Option C)
if uploaded_file:
    # Load & split
    try:
        docs = load_document_from_streamlit(uploaded_file)
        chunks = split_documents(docs)
        st.success(f"Loaded {len(docs)} pages → {len(chunks)} chunks.")
    except Exception:
        st.error("❌ Failed loading/splitting document.")
        st.code(traceback.format_exc())
        st.stop()

    # Vector store
    try:
        vector_store = create_vector_store(chunks)
        retriever = get_retriever(vector_store)
        st.info("Vector store created (in-memory Chroma).")
    except Exception:
        st.error("❌ Vector store creation failed.")
        st.code(traceback.format_exc())
        st.stop()

    # Inputs
    question = st.text_input("Enter your question:")
    profession = st.selectbox("Select profession for conclusion:", ["Engineer","Doctor","Lawyer","Student","Teacher","Developer"])

    if st.button("Generate Answer"):
        if not question.strip():
            st.warning("⚠ Please enter a question first.")
            st.stop()

        # Build RAG chain and run
        try:
            rag_chain = build_rag_chain(retriever, model_name=model_choice)
            rag_raw = rag_chain.invoke({"question": question})
            rag_text = extract_text(rag_raw)
        except Exception:
            st.error("❌ RAG chain failed.")
            st.code(traceback.format_exc())
            st.stop()

        # Profession conclusion
        try:
            pro_chain = build_profession_chain(model_name=model_choice)
            prof_raw = pro_chain.invoke({"profession": profession, "rag_answer": rag_text})
            prof_text = extract_text(prof_raw)
        except Exception:
            st.error("❌ Profession chain failed.")
            st.code(traceback.format_exc())
            st.stop()

        st.subheader("🟦 RAG Answer")
        st.write(rag_text)

        st.subheader(f"🟩 Conclusion ({profession})")
        st.write(prof_text)
