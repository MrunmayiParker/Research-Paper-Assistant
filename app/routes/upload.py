# from flask import Blueprint, request, jsonify, current_app
# from flask_login import login_required, current_user
# from werkzeug.utils import secure_filename
# import os
# from app.models.paper import Paper
# from rag.create_memory_llm import create_vector_store
# import hashlib
# from datetime import datetime, timezone


# upload_bp = Blueprint("upload", __name__, url_prefix="/upload")

# ALLOWED_EXTENSIONS = {"pdf"}

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def compute_file_hash(file):
#     file.seek(0)
#     content = file.read()
#     file.seek(0)  # Reset file pointer for saving
#     return hashlib.sha256(content).hexdigest()

# @upload_bp.route("/paper", methods=["POST"])
# @login_required
# def upload_paper():
#     from app import db

#     if 'file' not in request.files:
#         return jsonify({"message": "No file part"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"message": "No selected file"}), 400

#     file_hash = compute_file_hash(file)

#     # ✅ Check if this user already uploaded this paper
#     user_paper = Paper.query.filter_by(hash=file_hash, user_id=current_user.id).first()
#     if user_paper:
#         return jsonify({
#             "message": "Paper already uploaded.",
#             "paper_id": user_paper.id
#         }), 200

#     # ✅ Check if vectorstore for this hash exists (regardless of user)
#     existing_paper_any_user = Paper.query.filter_by(hash=file_hash).first()

#     # 🔥 Save file temporarily to disk
#     filename = secure_filename(file.filename)
#     filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
#     file.seek(0)  # Rewind file pointer before saving
#     file.save(filepath)

#     # 🔥 Only create vector store if it's missing
#     vectorstore_path = os.path.join("vectordb", file_hash)
#     if not os.path.exists(vectorstore_path):
#         create_vector_store(filepath, vectorstore_path)

#     # ✅ Save new paper entry for this user, reusing same file hash
#     paper = Paper(
#         filename=filename,
#         filepath=filepath,
#         upload_date=datetime.now(timezone.utc),
#         user_id=current_user.id,
#         hash=file_hash
#     )
#     db.session.add(paper)
#     db.session.commit()

#     return jsonify({
#         "message": "File uploaded successfully",
#         "paper_id": paper.id
#     }), 201

# # @upload_bp.route("/paper", methods=["POST"])
# # @login_required
# # def upload_paper():
# #     from app import db

# #     if 'file' not in request.files:
# #         return jsonify({"message": "No file part"}), 400

# #     file = request.files['file']
# #     if file.filename == '':
# #         return jsonify({"message": "No selected file"}), 400
    
# #     file_hash = compute_file_hash(file)

# #     # 🔥 Check if paper already exists
# #     existing = Paper.query.filter_by(hash=file_hash).first()
# #     if existing:
# #         return jsonify({
# #             "message": "Paper already exists",
# #             "paper_id": existing.id
# #         }), 200

# #     if file and allowed_file(file.filename):
# #         filename = secure_filename(file.filename)
# #         filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
# #         file.save(filepath)

# #         # 🔥 Use file_hash instead of filename to name the vectorstore folder
# #         vectorstore_path = os.path.join("vectordb", file_hash)

# #         if not os.path.exists(vectorstore_path):
# #             create_vector_store(filepath, vectorstore_path)

# #         paper = Paper(
# #             filename=filename,
# #             filepath=filepath,
# #             upload_date=datetime.now(timezone.utc),
# #             user_id=current_user.id,
# #             hash=file_hash
# #         )
# #         db.session.add(paper)
# #         db.session.commit()

# #         return jsonify({"message": "File uploaded successfully", "paper_id": paper.id}), 201

# #     return jsonify({"message": "Invalid file type"}), 400


# new upload ---------------------------------------------------------------

from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
from app.models.paper import Paper
from rag.create_memory_llm import create_vector_store
import hashlib
from datetime import datetime, timezone
from nlp_new.pipeline import process_and_recommend  # ✅

upload_bp = Blueprint("upload", __name__, url_prefix="/upload")

ALLOWED_EXTENSIONS = {"pdf"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compute_file_hash(file):
    file.seek(0)
    content = file.read()
    file.seek(0)  # Reset file pointer for saving
    return hashlib.sha256(content).hexdigest()

@upload_bp.route("/paper", methods=["POST"])
@login_required
def upload_paper():
    from app import db

    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    file_hash = compute_file_hash(file)

    user_paper = Paper.query.filter_by(hash=file_hash, user_id=current_user.id).first()
    if user_paper:
        return jsonify({
            "message": "Paper already uploaded.",
            "paper_id": user_paper.id
        }), 200

    filename = secure_filename(file.filename)
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.seek(0)
    file.save(filepath)

    vectorstore_path = os.path.join("vectordb", file_hash)
    if not os.path.exists(vectorstore_path):
        create_vector_store(filepath, vectorstore_path)

    paper = Paper(
        filename=filename,
        filepath=filepath,
        upload_date=datetime.now(timezone.utc),
        user_id=current_user.id,
        hash=file_hash
    )
    db.session.add(paper)
    db.session.commit()

    # ✅ Trigger classification + recommendation logging
    process_and_recommend(filepath, current_user.id, db.session)

    return jsonify({
        "message": "File uploaded successfully",
        "paper_id": paper.id
    }), 201
