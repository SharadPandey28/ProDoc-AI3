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
        chunk_overlap=150
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
# VECTOR STORE — CHROMA VERSION (COMPATIBLE WITH STREAMLIT CLOUD)
# ============================================================
from langchain_community.vectorstores import Chroma
import chromadb

def create_vector_store(chunks):
    model = get_embeddings_model()

    class Embeddings:
        def embed_documents(self, texts):
            return model.encode(texts).tolist()
        def embed_query(self, text):
            return model.encode([text])[0].tolist()

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
    return vector_store.as_retriever(
        search_kwargs={"k": k}
    )


# ============================================================
# RAG CHAIN
# ============================================================
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

def build_rag_chain(retriever):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Use ONLY the context to answer the question.

If no context is available, summarize the entire document.

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
            "question": RunnablePassthrough()
        })
        |
        (lambda x: {
            "context": "\n\n".join(doc.page_content for doc in x["context"]) if x["context"] else "NO_CONTEXT",
            "question": x["question"]
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
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    prompt = PromptTemplate(
        input_variables=["profession", "rag_answer"],
        template="""Write a conclusion from the perspective of a {profession}.  
Base it ONLY on the document answer below.

DOCUMENT ANSWER:
{rag_answer}

CONCLUSION ({profession}):
"""
    )

    return prompt | llm


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="RAG Document Analyzer — Chroma Version", layout="wide")
st.title("📄 RAG Document Analyzer (ChromaDB Version — Works on Streamlit Cloud)")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])

if uploaded_file:
    # -------------------------------------------
    # Load & Split Document
    # -------------------------------------------
    try:
        docs = load_document_from_streamlit(uploaded_file)
        chunks = split_documents(docs)
        st.success(f"Loaded {len(docs)} pages → {len(chunks)} chunks.")
    except:
        st.error("❌ Failed loading/splitting document.")
        st.code(traceback.format_exc())
        st.stop()

    # -------------------------------------------
    # Build Vector Store
    # -------------------------------------------
    try:
        vector_store = create_vector_store(chunks)
        retriever = get_retriever(vector_store)
    except:
        st.error("❌ Vector store creation failed.")
        st.code(traceback.format_exc())
        st.stop()

    # -------------------------------------------
    # Inputs
    # -------------------------------------------
    question = st.text_input("Enter your question:")
    profession = st.selectbox(
        "Select profession:",
        ["Engineer", "Doctor", "Lawyer", "Student", "Teacher", "Developer"]
    )

    # -------------------------------------------
    # Generate Answer
    # -------------------------------------------
    if st.button("Generate Answer"):
        if not question.strip():
            st.warning("⚠ Enter a question first.")
            st.stop()

        try:
            rag_chain = build_rag_chain(retriever)
            rag_resp = rag_chain.invoke({"question": question})
        except:
            st.error("❌ RAG failed.")
            st.code(traceback.format_exc())
            st.stop()

        try:
            pro_chain = build_profession_chain()
            conclusion = pro_chain.invoke({
                "profession": profession,
                "rag_answer": rag_resp.content
            })
        except:
            st.error("❌ Profession chain failed.")
            st.code(traceback.format_exc())
            st.stop()

        # -------------------------------------------
        # OUTPUT
        # -------------------------------------------
        st.subheader("🟦 RAG Answer")
        st.write(rag_resp.content)

        st.subheader(f"🟩 Conclusion ({profession})")
        st.write(conclusion.content)
