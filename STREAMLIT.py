# STREAMLIT.py — FULL WORKING SINGLE FILE (use this)
import streamlit as st
import tempfile
import traceback

# ------------------------
# DOCUMENT LOADER
# ------------------------
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

def load_document_from_streamlit(uploaded_file):
    if uploaded_file is None:
        raise ValueError("No file uploaded.")
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


# ------------------------
# TEXT SPLITTER
# ------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)


# ------------------------
# EMBEDDINGS (SentenceTransformer)
# ------------------------
from sentence_transformers import SentenceTransformer

@st.cache_resource
def get_embeddings_model():
    # loads once per session / worker
    return SentenceTransformer("all-MiniLM-L6-v2")


# ------------------------
# VECTOR STORE — use official langchain FAISS.from_documents
# ------------------------
# Use the official langchain vectorstore which expects an "embeddings" object with embed_documents / embed_query
from langchain.vectorstores import FAISS  # official wrapper
from langchain.schema import Document

def create_vector_store(chunks):
    embed_model = get_embeddings_model()

    class EmbedWrap:
        # embed_documents expects list[str] -> list[list[float]]
        def embed_documents(self, texts):
            # SentenceTransformer returns numpy array; convert to list of lists
            return embed_model.encode(texts).tolist()

        # embed_query expects a single string -> list[float]
        def embed_query(self, text):
            return embed_model.encode([text])[0].tolist()

    embeddings = EmbedWrap()

    # FAISS.from_documents expects an iterable of langchain Document objects.
    # chunks should already be Document objects from loader + splitter.
    faiss_index = FAISS.from_documents(chunks, embeddings)
    return faiss_index


# ------------------------
# RETRIEVER
# ------------------------
def get_retriever(vector_store, k=5):
    return vector_store.as_retriever(search_kwargs={"k": k})


# ------------------------
# RAG CHAIN
# ------------------------
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
        template=(
            "If context is 'NO_CONTEXT', summarize the full document.\n"
            "Otherwise answer using the context.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
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


# ------------------------
# PROFESSION CHAIN
# ------------------------
def build_profession_chain():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    prompt = PromptTemplate(
        input_variables=["profession", "rag_answer"],
        template=(
            "You are an expert {profession}.\n"
            "Write a concise conclusion from the {profession}'s perspective.\n\n"
            "Document Answer:\n{rag_answer}\n\n"
            "Conclusion:"
        )
    )
    return prompt | llm


# ------------------------
# STREAMLIT UI
# ------------------------
st.set_page_config(page_title="RAG Document Analyzer", layout="wide")
st.title("📄 RAG Document Analyzer")

uploaded_file = st.file_uploader("Upload your document", type=["pdf", "docx", "txt"])

if uploaded_file:
    try:
        docs = load_document_from_streamlit(uploaded_file)
        chunks = split_documents(docs)
        st.info(f"Loaded {len(docs)} source pages → {len(chunks)} chunks")
    except Exception as e:
        st.error("Failed to load/split document.")
        st.text(traceback.format_exc())
        raise

    try:
        vector_store = create_vector_store(chunks)
        retriever = get_retriever(vector_store)
    except Exception as e:
        st.error("Failed to create vector store or retriever. See console/logs for full traceback.")
        st.text(traceback.format_exc())
        raise

    question = st.text_input("Enter your question:")
    profession = st.selectbox("Select profession:", ["Engineer", "Doctor", "Lawyer", "Student", "Teacher", "Developer"])

    if st.button("Generate Answer"):
        if not question or question.strip() == "":
            st.warning("Please enter a question.")
        else:
            try:
                rag_chain = build_rag_chain(retriever)
                rag_response = rag_chain.invoke({"question": question})
            except Exception as e:
                st.error("RAG invocation failed. Full traceback below:")
                st.text(traceback.format_exc())
                raise

            try:
                pro_chain = build_profession_chain()
                final_conclusion = pro_chain.invoke({
                    "profession": profession,
                    "rag_answer": rag_response.content
                })
            except Exception as e:
                st.error("Profession chain failed. Full traceback below:")
                st.text(traceback.format_exc())
                raise

            st.subheader("RAG Answer")
            st.write(rag_response.content)

            st.subheader(f"Conclusion ({profession})")
            st.write(final_conclusion.content)
