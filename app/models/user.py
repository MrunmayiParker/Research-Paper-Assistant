# from app import db, bcrypt, login_manager
# from flask_login import UserMixin

# # @login_manager.user_loader
# # def load_user(user_id): # change
# #     return User.query.get(int(user_id))

# class User(db.Model, UserMixin):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(150), unique=True, nullable=False)
#     password = db.Column(db.String(150), nullable=False)
#     email = db.Column(db.String(150), unique=True, nullable=False)

#     def set_password(self, password):
#         self.password = bcrypt.generate_password_hash(password).decode('utf-8')
    
#     def check_password(self, password):
#         return bcrypt.check_password_hash(self.password, password)


# new user ------------------------------------

from app import db, bcrypt  # ✅ Import bcrypt from app
from flask_login import UserMixin
from datetime import datetime

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)

    # ✅ Add password helper methods
    def set_password(self, password):
        self.password = bcrypt.generate_password_hash(password).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.check_password_hash(self.password, password)

class PaperUpload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.Text, nullable=False)
    abstract = db.Column(db.Text, nullable=False)
    label = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
