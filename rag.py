import os
import nltk

os.environ["TOKENIZERS_PARALLELISM"] = "false"

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from dotenv import load_dotenv

load_dotenv()

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def build_qa_chain(pdf_paths: list):
    all_chunks = []

    # 1. Load all PDFs
    for path, filename in pdf_paths:
        loader = PyMuPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["filename"] = filename

        # Semantic chunking with NLTK
        splitter = NLTKTextSplitter(chunk_size=1000)
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    # 2. Local embeddings
    embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

    # 3. FAISS retriever (semantic)
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 4. BM25 retriever (keyword)
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = 4

    # 5. Hybrid retriever (50% FAISS + 50% BM25)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    # 6. Groq LLM
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # 7. History-aware retriever
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and latest question, reformulate the question to be standalone. Do not answer it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, hybrid_retriever, contextualize_prompt)

    # 8. QA chain
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer based only on the context below.
For summary requests, synthesize key points from the context.
Always mention the filename and page number where you found the answer.
If the context doesn't contain enough information, say so clearly.

Context: {context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # 9. Wrap with memory
    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_chain, vectorstore