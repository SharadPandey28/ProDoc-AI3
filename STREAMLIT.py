import streamlit as st
import tempfile

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
# EMBEDDINGS
# ------------------------
from sentence_transformers import SentenceTransformer

def get_embeddings():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ------------------------
# VECTOR STORE (FIXED)
# ------------------------
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks):
    embed_model = get_embeddings()

    class EmbedWrap:
        def embed_documents(self, texts):
            return embed_model.encode(texts).tolist()

        def embed_query(self, text):
            return embed_model.encode([text])[0].tolist()

    embedder = EmbedWrap()
    return FAISS.from_documents(chunks, embedder)


# ------------------------
# RETRIEVER
# ------------------------
def get_retriever(vector_store, k=5):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


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
            "context": "\n\n".join(doc.page_content for doc in x["context"])
            if x["context"] else "NO_CONTEXT",
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
            "Write a conclusion from the perspective of a {profession}.\n\n"
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
    docs = load_document_from_streamlit(uploaded_file)
    chunks = split_documents(docs)
    vector_store = create_vector_store(chunks)
    retriever = get_retriever(vector_store)

    question = st.text_input("Enter your question:")

    profession = st.selectbox(
        "Select profession:",
        ["Engineer", "Doctor", "Lawyer", "Student", "Teacher", "Developer"]
    )

    if st.button("Generate Answer"):
        rag_chain = build_rag_chain(retriever)
        rag_response = rag_chain.invoke({"question": question})

        pro_chain = build_profession_chain()
        final_conclusion = pro_chain.invoke({
            "profession": profession,
            "rag_answer": rag_response.content
        })

        st.subheader("RAG Answer")
        st.write(rag_response.content)

        st.subheader(f"Conclusion ({profession})")
        st.write(final_conclusion.content)
