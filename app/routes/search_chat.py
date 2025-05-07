# # app/routes/search_chat.py

# from flask import Blueprint, request, jsonify, render_template
# from app.routes.search import load_vector_store  # ✅ import here, reuse it
# from dotenv import load_dotenv
# import os
# from langchain.chains import RetrievalQA
# from langchain.llms import HuggingFaceEndpoint

# load_dotenv()

# search_ui_bp = Blueprint('search_ui', __name__, url_prefix='/search')

# @search_ui_bp.route('/chat', methods=['GET'])
# def search_chat_page():
#     return render_template('search_chat.html')

# @search_ui_bp.route('/ask', methods=['POST'])
# def search_ask_question():
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

#     vector_store = load_vector_store()  # ✅ reused from search.py
#     retriever = vector_store.as_retriever(search_type="similarity", k=4)

#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=True
#     )

#     response = qa(question)

#     return jsonify({
#         "answer": response["result"],
#         "sources": [doc.metadata for doc in response["source_documents"]]
#     }), 200

# app/routes/search_chat.py

from flask import Blueprint, request, jsonify, render_template
from app.routes.search import load_faiss_index  # ✅ updated to FAISS loader
from dotenv import load_dotenv
import os
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI  # ✅ OpenAI LLM

load_dotenv()

search_ui_bp = Blueprint('search_ui', __name__, url_prefix='/search')

@search_ui_bp.route('/chat', methods=['GET'])
def search_chat_page():
    return render_template('search_chat.html')

@search_ui_bp.route('/ask', methods=['POST'])
def search_ask_question():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"message": "Missing question"}), 400

    # ✅ Use OpenAI's GPT model
    llm = ChatOpenAI(
        temperature=0.6,
        model_name="gpt-4.1-nano-2025-04-14",  # Use gpt-4 if needed
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=256
    )

    # ✅ Use FAISS + OpenAI embeddings
    vector_store = load_faiss_index()
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})

    # Run QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa.invoke({"query": question})

    return jsonify({
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]]
    }), 200
