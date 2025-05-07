# # app/routes/home.py

# from flask import Blueprint, render_template
# from flask_login import login_required

# main_bp = Blueprint('main', __name__)

# @main_bp.route('/')
# def home():
#     return render_template('home.html')

# @main_bp.route('/search')
# @login_required
# def search():
#     return render_template('search.html')

# @main_bp.route('/chat')
# @login_required
# def chat():
#     return render_template('chat.html')

# @main_bp.route('/upload')
# @login_required
# def upload():
#     return render_template('upload.html')


# new home ----------------------------------------------------------------


from flask import Blueprint, render_template, redirect, url_for
from flask_login import login_required, current_user
from app.models.user import PaperUpload
from nlp_new.pipeline import get_dominant_label
from nlp_new.search_paper2 import search_papers
from app import db

main_bp = Blueprint('main', __name__)

# ✅ Redirect root to login or home based on login status
@main_bp.route('/')
def landing():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    else:
        return redirect(url_for('auth.login'))

@main_bp.route('/home')
@login_required
def home():
    user_id = current_user.id
    dominant_label = get_dominant_label(user_id, db.session)
    recommendations = []

    if dominant_label:
        recommendations = search_papers(dominant_label, top_k=5)

    return render_template('home.html', recs=recommendations)

@main_bp.route('/search')
@login_required
def search():
    return render_template('search.html')

@main_bp.route('/chat')
@login_required
def chat():
    return render_template('chat.html')

@main_bp.route('/upload')
@login_required
def upload():
    return render_template('upload.html')
