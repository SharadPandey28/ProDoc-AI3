# STREAMLIT.py — Final (Chroma + retry + gpt-4o-mini FIXED)
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
    return SentenceTransformer("all-MiniLM-L6-v2")


# ------------------------------
# Chroma vector store
# ------------------------------
from langchain_community.vectorstores import Chroma

def create_vector_store(chunks):
    model = get_embeddings_model()

    class Embedder:
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [v.tolist() for v in model.encode(texts)]

        def embed_query(self, text: str) -> List[float]:
            return model.encode(text).tolist()

    embeddings = Embedder()

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_docs"
    )


# ------------------------------
# Retriever
# ------------------------------
def get_retriever(vector_store, k=5):
    return vector_store.as_retriever(search_kwargs={"k": k})


# ------------------------------
# Raw HTTP OpenAI LLM (with retry)
# ------------------------------
from langchain.llms.base import LLM

class RawOpenAIChatLLM(LLM):
    api_key: str
    model: str = "gpt-4o-mini"    # FIXED

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", **kwargs):
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
                data = resp.json()
            except Exception as e:
                if attempt == max_attempts:
                    return f"Network error: {e}"
                time.sleep(min(20, 2 ** attempt))
                continue

            # Success
            if resp.status_code == 200 and "choices" in data:
                return data["choices"][0]["message"]["content"]

            # Rate limit
            if resp.status_code == 429:
                if attempt == max_attempts:
                    return json.dumps(data, indent=2)
                time.sleep(min(20, 2 ** attempt + random.random()))
                continue

            # Other errors
            return json.dumps(data, indent=2)

        return "❌ Unknown error"

    @property
    def _llm_type(self):
        return "raw_openai_chat"

    @property
    def _identifying_params(self):
        return {"model": self.model}


# ------------------------------
# Extract text helper
# ------------------------------
def extract_text(obj):
    if isinstance(obj, str):
        return obj
    try:
        return getattr(obj, "content", str(obj))
    except:
        return str(obj)


# ------------------------------
# RAG Chain
# ------------------------------
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

def build_rag_chain(retriever):
    llm = RawOpenAIChatLLM(api_key=st.secrets["OPENAI_API_KEY"])

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use ONLY the context to answer the question.  
If no context exists, summarize the entire document.

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
# Profession Chain
# ------------------------------
def build_profession_chain():
    llm = RawOpenAIChatLLM(api_key=st.secrets["OPENAI_API_KEY"])

    prompt = PromptTemplate(
        input_variables=["profession", "rag_answer"],
        template="""
Write a short conclusion from the perspective of a {profession}.
Use ONLY this answer:

{rag_answer}

CONCLUSION:
"""
    )

    return prompt | llm


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="RAG Analyzer", layout="wide")
st.title("📄 RAG Document Analyzer — FIXED MODEL")

uploaded_file = st.file_uploader("Upload document", type=["pdf", "docx", "txt"])

if uploaded_file:
    # Load & split
    try:
        docs = load_document_from_streamlit(uploaded_file)
        chunks = split_documents(docs)
        st.success(f"Loaded {len(docs)} pages → {len(chunks)} chunks.")
    except:
        st.error("❌ Error loading document")
        st.code(traceback.format_exc())
        st.stop()

    # Vector store
    try:
        vector_store = create_vector_store(chunks)
        retriever = get_retriever(vector_store)
    except:
        st.error("❌ Vector store error")
        st.code(traceback.format_exc())
        st.stop()

    question = st.text_input("Ask your question:")
    profession = st.selectbox(
        "Select profession:",
        ["Engineer", "Doctor", "Lawyer", "Student", "Teacher", "Developer"]
    )

    if st.button("Generate Answer"):
        # RAG
        try:
            rag_chain = build_rag_chain(retriever)
            rag_raw = rag_chain.invoke({"question": question})
            rag_text = extract_text(rag_raw)
        except:
            st.error("❌ RAG Error")
            st.code(traceback.format_exc())
            st.stop()

        # Profession
        try:
            pro_chain = build_profession_chain()
            pro_raw = pro_chain.invoke({"profession": profession, "rag_answer": rag_text})
            pro_text = extract_text(pro_raw)
        except:
            st.error("❌ Profession Error")
            st.code(traceback.format_exc())
            st.stop()

        st.subheader("🟦 RAG Answer")
        st.write(rag_text)

        st.subheader(f"🟩 Conclusion ({profession})")
        st.write(pro_text)
