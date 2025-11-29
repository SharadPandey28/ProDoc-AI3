import streamlit as st
import tempfile
import traceback
import uuid
import time

# ===================================================================
# DOCUMENT LOADER
# ===================================================================
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
        raise ValueError("Unsupported file type")

    return loader.load()


# ===================================================================
# TEXT SPLITTING
# ===================================================================
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)


# ===================================================================
# EMBEDDINGS
# ===================================================================
from sentence_transformers import SentenceTransformer

@st.cache_resource
def get_embeddings_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ===================================================================
# CHROMA VECTOR STORE (ISOLATED PER UPLOAD)
# ===================================================================
from langchain_community.vectorstores import Chroma

def create_vector_store(chunks):
    model = get_embeddings_model()

    class Embedder:
        def embed_documents(self, texts):
            return [v.tolist() for v in model.encode(texts)]

        def embed_query(self, text):
            return model.encode(text).tolist()

    embeddings = Embedder()

    # FIX: Unique collection per user upload → prevents mixed PDF data
    session_id = str(uuid.uuid4())
    collection_name = f"rag_docs_{session_id}"

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=None,
    )

    return vector_store


# ===================================================================
# RETRIEVER
# ===================================================================
def get_retriever(vector_store, k=5):
    return vector_store.as_retriever(search_kwargs={"k": k})


# ===================================================================
# CUSTOM RAW OPENAI LLM (NO BUGS)
# ===================================================================
from langchain.llms.base import LLM
import requests
from typing import Optional, List

class RawOpenAIChatLLM(LLM):
    model: str = "gpt-4o-mini"
    max_retries: int = 5

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }

        for attempt in range(self.max_retries):
            try:
                res = requests.post(url, json=payload, headers=headers)
                if res.status_code == 429:  # rate limit
                    time.sleep(3)
                    continue
                res.raise_for_status()
                return res.json()["choices"][0]["message"]["content"]

            except Exception as e:
                if attempt == self.max_retries - 1:
                    return f"[OpenAI Error] {str(e)}"
                time.sleep(2)

        return "ERROR"

    @property
    def _identifying_params(self):
        return {"model": self.model}

    @property
    def _llm_type(self):
        return "raw_openai"


# ===================================================================
# RAG CHAIN
# ===================================================================
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

def build_rag_chain(retriever):
    llm = RawOpenAIChatLLM(model="gpt-4o-mini")

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Answer the question ONLY using the context below.

If context is empty, summarize the whole document.

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
        | (lambda x: {
            "context": "\n\n".join(doc.page_content for doc in x["context"]) if x["context"] else "",
            "question": x["question"],
        })
        | prompt
        | llm
    )

    return chain


# ===================================================================
# PROFESSION CHAIN
# ===================================================================
def build_profession_chain():
    llm = RawOpenAIChatLLM(model="gpt-4o-mini")

    prompt = PromptTemplate(
        input_variables=["profession", "rag_answer"],
        template="""
Write a conclusion from the perspective of a {profession}.
Use ONLY the answer below.

ANSWER:
{rag_answer}

CONCLUSION:
"""
    )

    return prompt | llm


# ===================================================================
# STREAMLIT UI
# ===================================================================
st.set_page_config(page_title="RAG Analyzer — FINAL", layout="wide")
st.title("📄 AI RAG Document Analyzer — FINAL FIXED BUILD 🔥")

uploaded_file = st.file_uploader("Upload PDF, DOCX or TXT", type=["pdf", "docx", "txt"])

if uploaded_file:
    # LOAD DOC
    try:
        docs = load_document_from_streamlit(uploaded_file)
        chunks = split_documents(docs)
        st.success(f"Loaded {len(docs)} pages → {len(chunks)} chunks.")
    except:
        st.error("❌ Failed to load or split document.")
        st.code(traceback.format_exc())
        st.stop()

    # BUILD VECTORSTORE
    try:
        vector_store = create_vector_store(chunks)
        retriever = get_retriever(vector_store)
    except:
        st.error("❌ Failed building vector store.")
        st.code(traceback.format_exc())
        st.stop()

    question = st.text_input("Enter your question:")
    profession = st.selectbox(
        "Generate conclusion as:",
        ["Engineer", "Doctor", "Lawyer", "Developer", "Teacher", "Student"]
    )

    if st.button("Generate Answer"):
        if not question.strip():
            st.warning("Enter a question first.")
            st.stop()

        # RAG
        try:
            rag_chain = build_rag_chain(retriever)
            rag_answer = rag_chain.invoke({"question": question})
        except:
            st.error("❌ RAG failed.")
            st.code(traceback.format_exc())
            st.stop()

        # Profession Output
        try:
            pro_chain = build_profession_chain()
            conclusion = pro_chain.invoke({
                "profession": profession,
                "rag_answer": rag_answer
            })
        except:
            st.error("❌ Profession conclusion failed.")
            st.code(traceback.format_exc())
            st.stop()

        st.subheader("🔵 RAG Answer")
        st.write(rag_answer)

        st.subheader(f"🟢 Conclusion ({profession})")
        st.write(conclusion)
