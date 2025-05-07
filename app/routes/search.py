# # routes/search_chat.py

# from flask import Blueprint, request, jsonify, current_app, render_template
# import os
# import time
# import requests
# import fitz  # PyMuPDF
# import feedparser
# from urllib.parse import quote
# from io import BytesIO
# from uuid import uuid4

# from dotenv import load_dotenv
# from langchain.embeddings import HuggingFaceBgeEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone, ServerlessSpec
# from langchain.chains import RetrievalQA
# from langchain.llms import HuggingFaceEndpoint

# # Load environment variables
# load_dotenv()

# # Blueprint Setup
# search_bp = Blueprint('search', __name__, url_prefix='/api')

# # ----------------- Helper Functions -----------------
# def load_vector_store():
#     embeddings = setup_embeddings()
#     index = connect_to_pinecone()
#     return PineconeVectorStore(index=index, embedding=embeddings)

# def connect_to_pinecone():
#     pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#     index_name = os.getenv("PINECONE_INDEX", "rpapers")

#     existing_indexes = [i['name'] for i in pc.list_indexes()]
#     if index_name not in existing_indexes:
#         pc.create_index(
#             name=index_name,
#             dimension=384,
#             metric="cosine",
#             spec=ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_ENV"))
#         )

#     return pc.Index(index_name)

# def extract_text_from_pdf_url(pdf_url):
#     response = requests.get(pdf_url)
#     with fitz.open("pdf", BytesIO(response.content)) as doc:
#         return "\n".join(page.get_text() for page in doc)

# def split_text_into_chunks(text):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=240)
#     return splitter.create_documents([text])

# def setup_embeddings():
#     return HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")

# def namespace_has_data(index, namespace="default"):
#     """Returns True if the Pinecone namespace contains any data."""
#     try:
#         response = index.describe_index_stats()
#         ns_info = response.get("namespaces", {})
#         return namespace in ns_info and ns_info[namespace]["vector_count"] > 0
#     except Exception as e:
#         print("⚠️ Error checking namespace:", e)
#         return False

# # ----------------- Routes -----------------

# @search_bp.route('/search_papers', methods=['POST'])
# def search_papers():
#     """Search Arxiv papers based on query."""
#     data = request.get_json()
#     query = data.get('query')
#     max_results = data.get('max_results', 5)

#     if not query:
#         return jsonify({"message": "Missing search query"}), 400

#     base_url = "http://export.arxiv.org/api/query?"
#     search_url = f"{base_url}search_query=all:{quote(query)}&start=0&max_results={max_results}"
#     feed = feedparser.parse(search_url)

#     results = []
#     for entry in feed.entries:
#         pdf_url = next(link.href for link in entry.links if link.type == "application/pdf")
#         results.append({
#             "title": entry.title,
#             "summary": entry.summary,
#             "pdf_url": pdf_url
#         })

#     return jsonify(results), 200


# @search_bp.route('/prepare_paper', methods=['POST'])
# def prepare_paper():
#     data = request.get_json()
#     pdf_url = data.get('pdf_url')
#     namespace = "default"

#     if not pdf_url:
#         return jsonify({"message": "Missing PDF URL"}), 400

#     try:
#         pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#         index = pc.Index(os.getenv("PINECONE_INDEX", "rpapers"))

#         # ✅ Delete only if namespace has data
#         if namespace_has_data(index, namespace=namespace):
#             index.delete(delete_all=True, namespace=namespace)
#             print(f"🧹 Cleared existing namespace '{namespace}'")

#         # Extract, split, and embed new content
#         full_text = extract_text_from_pdf_url(pdf_url)
#         chunks = split_text_into_chunks(full_text)
#         vector_store = load_vector_store()

#         documents = [Document(page_content=chunk.page_content, metadata={"url": pdf_url}) for chunk in chunks]
#         uuids = [str(uuid4()) for _ in documents]

#         vector_store.add_documents(documents=documents, ids=uuids, namespace=namespace)

#         return jsonify({"message": "Paper processed and stored successfully."}), 200

#     except Exception as e:
#         print("❌ Exception in /prepare_paper:", e)
#         return jsonify({"message": str(e)}), 500



# @search_bp.route('/ask_question', methods=['POST'])
# def ask_question():
#     data = request.get_json()
#     question = data.get('question')

#     if not question:
#         return jsonify({"message": "Missing question"}), 400

#     llm = HuggingFaceEndpoint(
#         repo_id="mistralai/Mistral-7B-Instruct-v0.3",
#         temperature=0.6,
#         task="text-generation",
#         huggingfacehub_api_token=os.getenv("HF_TOKEN"),
#         max_new_tokens=512
#     )

#     vector_store = load_vector_store()

#     retriever = vector_store.as_retriever(search_type="similarity", k=4)

#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=True
#     )

#     response = qa(question)
#     print(response["result"])

#     return jsonify({
#         "answer": response["result"],
#         "sources": [doc.metadata for doc in response["source_documents"]]
#     }), 200

# search_ui_bp = Blueprint('search_ui', __name__, url_prefix='/search')

# @search_ui_bp.route('/chat')
# def search_chat_page():
#     return render_template('search_chat.html')

#==============================================================

# from flask import Blueprint, request, jsonify, current_app, render_template
# import os
# import time
# import requests
# import fitz  # PyMuPDF
# import feedparser
# from urllib.parse import quote
# from io import BytesIO
# from uuid import uuid4

# from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_core.prompts import PromptTemplate
# from langchain.docstore.document import Document
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone, ServerlessSpec
# from langchain.chains import RetrievalQA
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # ✅ OPENAI modules

# # Load environment variables
# load_dotenv()

# # Blueprint Setup
# search_bp = Blueprint('search', __name__, url_prefix='/api')

# # ----------------- Helper Functions -----------------
# def load_vector_store():
#     embeddings = setup_embeddings()
#     index = connect_to_pinecone()
#     return PineconeVectorStore(index=index, embedding=embeddings, namespace="default")

# def connect_to_pinecone():
#     pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#     index_name = os.getenv("PINECONE_INDEX", "rpapers")

#     existing_indexes = [i['name'] for i in pc.list_indexes()]
#     if index_name not in existing_indexes:
#         pc.create_index(
#             name=index_name,
#             dimension=1536,  # ✅ OpenAI Embeddings = 1536 dimensions
#             metric="cosine",
#             spec=ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_ENV"))
#         )

#     return pc.Index(index_name)

# def extract_text_from_pdf_url(pdf_url):
#     response = requests.get(pdf_url)
#     with fitz.open("pdf", BytesIO(response.content)) as doc:
#         return "\n".join(page.get_text() for page in doc)

# def split_text_into_chunks(text):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
#     return splitter.create_documents([text])

# def setup_embeddings():
#     return OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))  # ✅

# def namespace_has_data(index, namespace="default"):
#     """Returns True if the Pinecone namespace contains any data."""
#     try:
#         response = index.describe_index_stats()
#         ns_info = response.get("namespaces", {})
#         return namespace in ns_info and ns_info[namespace]["vector_count"] > 0
#     except Exception as e:
#         print("⚠️ Error checking namespace:", e)
#         return False

# # ----------------- Routes -----------------

# @search_bp.route('/search_papers', methods=['POST'])
# def search_papers():
#     """Search Arxiv papers based on query."""
#     data = request.get_json()
#     query = data.get('query')
#     max_results = data.get('max_results', 5)

#     if not query:
#         return jsonify({"message": "Missing search query"}), 400

#     base_url = "http://export.arxiv.org/api/query?"
#     search_url = f"{base_url}search_query=all:{quote(query)}&start=0&max_results={max_results}"
#     feed = feedparser.parse(search_url)

#     results = []
#     for entry in feed.entries:
#         pdf_url = next(link.href for link in entry.links if link.type == "application/pdf")
#         results.append({
#             "title": entry.title,
#             "summary": entry.summary,
#             "pdf_url": pdf_url
#         })

#     return jsonify(results), 200


# @search_bp.route('/prepare_paper', methods=['POST'])
# def prepare_paper():
#     data = request.get_json()
#     pdf_url = data.get('pdf_url')
#     namespace = "default"

#     if not pdf_url:
#         return jsonify({"message": "Missing PDF URL"}), 400

#     try:
#         pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#         index = pc.Index(os.getenv("PINECONE_INDEX", "rpapers"))

#         # ✅ Delete only if namespace has data
#         if namespace_has_data(index, namespace=namespace):
#             index.delete(delete_all=True, namespace=namespace)
#             print(f"🧹 Cleared existing namespace '{namespace}'")

#         # Extract, split, and embed new content
#         full_text = extract_text_from_pdf_url(pdf_url)
#         print("EXTRACTED TEXT SAMPLE:", full_text[:500])
#         chunks = split_text_into_chunks(full_text)
#         vector_store = load_vector_store()

#         documents = [Document(page_content=chunk.page_content, metadata={"url": pdf_url}) for chunk in chunks]
#         uuids = [str(uuid4()) for _ in documents]

#         vector_store.add_documents(documents=documents, ids=uuids, namespace=namespace)
#         time.sleep(4)
#         print("✅ Added", len(documents), "documents to Pinecone")

#         return jsonify({"message": "Paper processed and stored successfully."}), 200

#     except Exception as e:
#         print("❌ Exception in /prepare_paper:", e)
#         return jsonify({"message": str(e)}), 500



# from langchain_core.prompts import PromptTemplate

# @search_bp.route('/ask_question', methods=['POST'])
# def ask_question():
#     data = request.get_json()
#     question = data.get('question')

#     if not question:
#         return jsonify({"message": "Missing question"}), 400

#     # Initialize OpenAI LLM
#     llm = ChatOpenAI(
#         temperature=0.6,
#         model_name="gpt-4.1-nano-2025-04-14",
#         openai_api_key=os.getenv("OPENAI_API_KEY"),
#         max_tokens=256
#     )

#     # Load retriever with MMR (diverse, high-relevance chunks)
#     vector_store = load_vector_store()
#     retriever = vector_store.as_retriever(
#         search_type="mmr",
#         search_kwargs={"k": 3, "fetch_k": 10}
#     )

#     # Debug: Log retrieved chunks
#     docs = retriever.get_relevant_documents(question)
#     print(f"📥 Retrieved {len(docs)} chunks:")
#     for i, d in enumerate(docs):
#         section = d.metadata.get("section", "body")
#         print(f"{i+1}. [{section}] {d.page_content[:100]}...")

#     # Define prompt template for grounded answer
#     prompt = PromptTemplate(
#         template="""
#             Answer the question using only the context provided below.
#             This context is extracted from a research paper uploaded by the user.
#             Do not make up facts or speculate. If the answer is not in the context, say you don't know.

#             Context:
#             {context}

#             Question:
#             {question}

#             Answer:
#             """,
#         input_variables=["context", "question"]
#     )

#     # Build RetrievalQA chain with prompt
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type_kwargs={'prompt': prompt},
#         input_key="question"
#     )

#     # Invoke with proper input key
#     response = qa.invoke({"question": question})
#     print("🧠 Answer:", response["result"])

#     return jsonify({
#         "answer": response["result"],
#         "sources": [doc.metadata for doc in response["source_documents"]]
#     }), 200


# # Search UI (Optional Frontend Route)
# search_ui_bp = Blueprint('search_ui', __name__, url_prefix='/search')

# @search_ui_bp.route('/chat')
# def search_chat_page():
#     return render_template('search_chat.html')

# ===================================================================================


# @search_bp.route('/ask_question', methods=['POST'])
# def ask_question():
#     data = request.get_json()
#     question = data.get('question')

#     if not question:
#         return jsonify({"message": "Missing question"}), 400

#     llm = ChatOpenAI(  # ✅ OpenAI LLM
#         temperature=0.6,
#         model_name="gpt-4.1-nano-2025-04-14",  # Cheapest and works well for Q&A
#         openai_api_key=os.getenv("OPENAI_API_KEY"),
#         max_tokens=256
#     )

#     vector_store = load_vector_store()
#     retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={
#         "k": 3,          # return only top 3 to the LLM
#         "fetch_k": 10    # search pool size (kept small)
#     })

#     docs = retriever.get_relevant_documents(question)
#     print("📥 Retrieved chunks:", len(docs))
#     for d in docs:
#         print("-", d.page_content[:100])


#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=True
#     )

#     response = qa.invoke({"query": question})
#     print(response["result"])

#     return jsonify({
#         "answer": response["result"],
#         "sources": [doc.metadata for doc in response["source_documents"]]
#     }), 200

# ===================================================================================

from flask import Blueprint, request, jsonify, render_template
import os
import requests
import fitz  # PyMuPDF
from urllib.parse import quote
from io import BytesIO
import feedparser

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
import shutil
import time

# Load env variables
load_dotenv()

# Constants
FAISS_INDEX_PATH = "vectorstore_search/current_index"
PDF_STORAGE_PATH = "data/latest.pdf"

# Blueprint Setup
search_bp = Blueprint('search', __name__, url_prefix='/api')
search_ui_bp = Blueprint('search_ui', __name__, url_prefix='/search')

# ----------------- Helper Functions -----------------
def extract_text_from_pdf_url(pdf_url):
    response = requests.get(pdf_url)
    os.makedirs("data", exist_ok=True)
    with open(PDF_STORAGE_PATH, "wb") as f:
        f.write(response.content)

    with fitz.open(PDF_STORAGE_PATH) as doc:
        return "\n".join(page.get_text() for page in doc)

def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.create_documents([text])

def setup_embeddings():
    return OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

def save_faiss_index(chunks):
    embedding_model = setup_embeddings()
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)

    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(FAISS_INDEX_PATH)
    print(f"✅ FAISS index saved to {FAISS_INDEX_PATH}")

def load_faiss_index():
    embedding_model = setup_embeddings()
    return FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)

# ----------------- Routes -----------------

@search_bp.route('/search_papers', methods=['POST'])
def search_papers():
    """Search Arxiv papers based on query."""
    data = request.get_json()
    query = data.get('query')
    max_results = data.get('max_results', 5)

    if not query:
        return jsonify({"message": "Missing search query"}), 400

    base_url = "http://export.arxiv.org/api/query?"
    search_url = f"{base_url}search_query=all:{quote(query)}&start=0&max_results={max_results}"
    feed = feedparser.parse(search_url)

    results = []
    for entry in feed.entries:
        pdf_url = next(link.href for link in entry.links if link.type == "application/pdf")
        results.append({
            "title": entry.title,
            "summary": entry.summary,
            "pdf_url": pdf_url
        })

    return jsonify(results), 200

@search_bp.route('/prepare_paper', methods=['POST'])
def prepare_paper():
    data = request.get_json()
    pdf_url = data.get('pdf_url')

    if not pdf_url:
        return jsonify({"message": "Missing PDF URL"}), 400

    try:
        # Extract, chunk, embed, save
        full_text = extract_text_from_pdf_url(pdf_url)
        print("📄 Extracted Text Sample:\n", full_text[:400])
        chunks = split_text_into_chunks(full_text)
        save_faiss_index(chunks)

        return jsonify({"message": "Paper processed and FAISS index updated."}), 200
    except Exception as e:
        print("❌ Exception in /prepare_paper:", e)
        return jsonify({"message": str(e)}), 500


@search_bp.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"message": "Missing question"}), 400

    try:
        llm = ChatOpenAI(
            temperature=0.6,
            model_name="gpt-4.1-nano-2025-04-14",  # ✅ or gpt-4 if needed
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=256
        )

        db = load_faiss_index()
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        # Optional: inspect retrieved content
        docs = retriever.get_relevant_documents(question)
        print(f"📥 Retrieved {len(docs)} chunks:")
        for i, d in enumerate(docs):
            print(f"{i+1}. {d.page_content[:100]}...")

        qa_prompt = PromptTemplate.from_template("""
                You are an expert research assistant. Use only the following context extracted from a research paper to answer the user's question.

                If the answer is clearly stated, respond directly.
                If it requires reasoning, explain the logic.
                If the context does not contain enough information, say "I don't know."

                Context:
                {context}

                Question:
                {question}

                Answer:
                """)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_prompt},
        )

        result = qa(question)
        return jsonify({
            "answer": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        }), 200

    except Exception as e:
        print("❌ Exception in /ask_question:", e)
        return jsonify({"message": str(e)}), 500


@search_ui_bp.route('/chat')
def search_chat_page():
    return render_template('search_chat.html')
