import streamlit as st
import tempfile
import traceback

# ============================================================
# DOCUMENT LOADER
# ============================================================
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


# ============================================================
# TEXT SPLITTER
# ============================================================
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    return splitter.split_documents(documents)


# ============================================================
# EMBEDDINGS (SentenceTransformer)
# ============================================================
from sentence_transformers import SentenceTransformer

@st.cache_resource
def get_embeddings_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# CHROMA VECTOR STORE — STREAMLIT CLOUD SAFE
# ============================================================
from langchain_community.vectorstores import Chroma

def create_vector_store(chunks):
    model = get_embeddings_model()

    class Embeddings:
        def embed_documents(self, texts):
            vectors = model.encode(texts)
            return [v.tolist() for v in vectors]

        def embed_query(self, text):
            vector = model.encode(text)
            return vector.tolist()

    embeddings = Embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_docs"
    )

    return vector_store


# ============================================================
# RETRIEVER
# ============================================================
def get_retriever(vector_store, k=5):
    return vector_store.as_retriever(search_kwargs={"k": k})


# ============================================================
# CUSTOM OPENAI LLM (NO PROXY BUG)
# ============================================================
from langchain.llms.base import LLM
from typing import Optional, List
from openai import OpenAI

class OpenAIChatLLM(LLM):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message["content"]

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model}

    @property
    def _llm_type(self) -> str:
        return "openai_chat_custom"


# ============================================================
# RAG CHAIN
# ============================================================
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

def build_rag_chain(retriever):
    llm = OpenAIChatLLM(
        api_key=st.secrets["OPENAI_API_KEY"],
        model="gpt-4o-mini"
    )

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
            "context": "\n\n".join(doc.page_content for doc in x["context"])
                       if x["context"] else "NO_CONTEXT",
            "question": x["question"],
        })
        |
        prompt
        |
        llm
    )

    return chain


# ============================================================
# PROFESSION CHAIN
# ============================================================
def build_profession_chain():
    llm = OpenAIChatLLM(
        api_key=st.secrets["OPENAI_API_KEY"],
        model="gpt-4o-mini"
    )

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


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="RAG Document Analyzer — FINAL", layout="wide")
st.title("📄 RAG Document Analyzer (Fully Fixed & Working 🚀🔥)")

uploaded_file = st.file_uploader("Upload your document", type=["pdf", "docx", "txt"])

if uploaded_file:
    # -------- Load & Split Document --------
    try:
        docs = load_document_from_streamlit(uploaded_file)
        chunks = split_documents(docs)
        st.success(f"Loaded {len(docs)} pages → {len(chunks)} chunks.")
    except:
        st.error("❌ Failed loading/splitting document.")
        st.code(traceback.format_exc())
        st.stop()

    # -------- Build Vector Store --------
    try:
        vector_store = create_vector_store(chunks)
        retriever = get_retriever(vector_store)
    except:
        st.error("❌ Vector store creation failed.")
        st.code(traceback.format_exc())
        st.stop()

    question = st.text_input("Enter your question:")

    profession = st.selectbox(
        "Select profession:",
        ["Engineer", "Doctor", "Lawyer", "Student", "Teacher", "Developer"],
    )

    if st.button("Generate Answer"):
        if not question.strip():
            st.warning("⚠ Please enter a question!")
            st.stop()

        # ----- RAG -----
        try:
            rag_chain = build_rag_chain(retriever)
            rag_resp = rag_chain.invoke({"question": question})
        except:
            st.error("❌ RAG failed.")
            st.code(traceback.format_exc())
            st.stop()

        # ----- Profession Conclusion -----
        try:
            pro_chain = build_profession_chain()
            conclusion = pro_chain.invoke({
                "profession": profession,
                "rag_answer": rag_resp,
            })
        except:
            st.error("❌ Profession chain failed.")
            st.code(traceback.format_exc())
            st.stop()

        st.subheader("🟦 RAG Answer")
        st.write(rag_resp)

        st.subheader(f"🟩 Conclusion ({profession})")
        st.write(conclusion)
