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
# CHROMA VECTOR STORE
# ============================================================
from langchain_community.vectorstores import Chroma

def create_vector_store(chunks):
    model = get_embeddings_model()

    class Embedder:
        def embed_documents(self, texts):
            return model.encode(texts).tolist()

        def embed_query(self, text):
            return model.encode([text])[0].tolist()

    embeddings = Embedder()

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
# RAW OPENAI CHAT LLM (NO Pydantic / NO PROXY BUG)
# ============================================================
from langchain.llms.base import LLM
from typing import Optional, List

class RawOpenAIChatLLM(LLM):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        import requests, json

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            data = r.json()
        except Exception as e:
            return f"NETWORK ERROR: {e}"

        try:
            return data["choices"][0]["message"]["content"]
        except:
            return json.dumps(data, indent=2)

    @property
    def _llm_type(self):
        return "raw_openai_chat"

    @property
    def _identifying_params(self):
        return {"model": self.model}

# ============================================================
# RAG CHAIN
# ============================================================
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

def build_rag_chain(retriever):
    llm = RawOpenAIChatLLM(
        api_key=st.secrets["OPENAI_API_KEY"],
        model="gpt-4o-mini"
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use ONLY the context to answer the question.
If no context exists, summarize the document.

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
        | prompt
        | llm
    )

    return chain

# ============================================================
# PROFESSION CHAIN
# ============================================================
def build_profession_chain():
    llm = RawOpenAIChatLLM(
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
st.set_page_config(page_title="RAG Document Analyzer", layout="wide")
st.title("📄 FINAL RAG Document Analyzer (Working Build 🚀)")

uploaded_file = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"])

if uploaded_file:
    try:
        docs = load_document_from_streamlit(uploaded_file)
        chunks = split_documents(docs)
        st.success(f"Loaded {len(docs)} pages → {len(chunks)} chunks")
    except:
        st.error("❌ Failed loading/splitting document")
        st.code(traceback.format_exc())
        st.stop()

    try:
        vector_store = create_vector_store(chunks)
        retriever = get_retriever(vector_store)
    except:
        st.error("❌ Failed creating vector store")
        st.code(traceback.format_exc())
        st.stop()

    question = st.text_input("Enter your question:")
    profession = st.selectbox("Select profession view:", ["Engineer", "Doctor", "Lawyer", "Student", "Teacher", "Developer"])

    if st.button("Generate Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        try:
            rag_chain = build_rag_chain(retriever)
            rag_result = rag_chain.invoke({"question": question})
        except:
            st.error("❌ RAG query failed")
            st.code(traceback.format_exc())
            st.stop()

        try:
            pro_chain = build_profession_chain()
            conclusion = pro_chain.invoke({
                "profession": profession,
                "rag_answer": rag_result,
            })
        except:
            st.error("❌ Profession chain failed")
            st.code(traceback.format_exc())
            st.stop()

        st.subheader("🔵 RAG Answer")
        st.write(rag_result)

        st.subheader(f"🟢 Conclusion ({profession})")
        st.write(conclusion)
