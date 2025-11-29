import streamlit as st
import tempfile
import traceback
import requests
import json
from typing import Optional, List

# ---------------------------
# Document loaders
# ---------------------------
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


# ---------------------------
# Text splitter
# ---------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    return splitter.split_documents(documents)


# ---------------------------
# Embeddings model
# ---------------------------
from sentence_transformers import SentenceTransformer

@st.cache_resource
def get_embeddings_model():
    # model downloads on first run — stable and small
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------
# Chroma vector store
# ---------------------------
from langchain_community.vectorstores import Chroma

def create_vector_store(chunks):
    model = get_embeddings_model()

    class Embeddings:
        # returns List[List[float]]
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            vectors = model.encode(texts)
            return [v.tolist() for v in vectors]

        # returns List[float]
        def embed_query(self, text: str) -> List[float]:
            vec = model.encode(text)
            # model.encode(text) returns a 1-D vector for single string
            return vec.tolist()

    embeddings = Embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_docs"
    )
    return vector_store


# ---------------------------
# Retriever
# ---------------------------
def get_retriever(vector_store, k: int = 5):
    return vector_store.as_retriever(search_kwargs={"k": k})


# ---------------------------
# Raw HTTP OpenAI LLM (Pydantic-compatible)
# ---------------------------
# LangChain LLM base in your version uses pydantic. To be safe we declare fields using pydantic.
from langchain.llms.base import LLM
try:
    # pydantic v1 / v2 compatible import
    from pydantic import BaseModel, Field
except Exception:
    # fallback, but pydantic should exist in your environment
    from pydantic import BaseModel, Field

class RawOpenAIChatLLM(LLM, BaseModel):
    api_key: str = Field(...)
    model: str = Field(default="gpt-4o-mini")

    class Config:
        arbitrary_types_allowed = True

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        # simple POST call; handle network errors gracefully
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            data = resp.json()
        except Exception as e:
            return f"API/network error: {e}"

        # try to extract text in common shapes
        try:
            # standard OpenAI chat response
            return data["choices"][0]["message"]["content"]
        except Exception:
            # fallback: return whole json for debugging
            return "API Error — unexpected response:\n" + json.dumps(data, indent=2)

    @property
    def _llm_type(self) -> str:
        return "raw_openai_chat"

    @property
    def _identifying_params(self):
        return {"model": self.model}


# ---------------------------
# Helpers to safely extract text from runnable outputs
# ---------------------------
def extract_text(obj) -> str:
    """
    LangChain runnable.invoke can return different shapes.
    This helper tries common possibilities and returns a plain string.
    """
    if obj is None:
        return ""
    # If it's already a string
    if isinstance(obj, str):
        return obj
    # If it's a LangChain message-like with 'content'
    try:
        content = getattr(obj, "content", None)
        if isinstance(content, str):
            return content
    except Exception:
        pass
    # If object has 'text' or 'output' keys
    if isinstance(obj, dict):
        for key in ("content", "text", "output"):
            if key in obj and isinstance(obj[key], str):
                return obj[key]
    # If it's a pydantic model or has .__dict__, try to stringify
    try:
        s = str(obj)
        return s
    except Exception:
        return ""


# ---------------------------
# RAG chain
# ---------------------------
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

def build_rag_chain(retriever):
    llm = RawOpenAIChatLLM(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4o-mini")

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


# ---------------------------
# Profession conclusion chain
# ---------------------------
def build_profession_chain():
    llm = RawOpenAIChatLLM(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4o-mini")

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


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="RAG Document Analyzer", layout="wide")
st.title("📄 RAG Document Analyzer (Final)")

uploaded_file = st.file_uploader("Upload your document", type=["pdf", "docx", "txt"])

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
    except Exception:
        st.error("❌ Vector store creation failed.")
        st.code(traceback.format_exc())
        st.stop()

    # Inputs
    question = st.text_input("Enter your question:")
    profession = st.selectbox("Select profession for conclusion:", ["Engineer","Doctor","Lawyer","Student","Teacher","Developer"])

    if st.button("Generate Answer"):
        if not question.strip():
            st.warning("⚠ Please enter a question.")
            st.stop()

        # RAG
        try:
            rag_chain = build_rag_chain(retriever)
            rag_raw = rag_chain.invoke({"question": question})
            rag_text = extract_text(rag_raw)
        except Exception:
            st.error("❌ RAG failed.")
            st.code(traceback.format_exc())
            st.stop()

        # Profession conclusion
        try:
            pro_chain = build_profession_chain()
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
