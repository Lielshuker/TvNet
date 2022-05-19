from app import db


class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(256))
    first_name = db.Column(db.String(256))
    last_name = db.Column(db.String(256))
    email = db.Column(db.String(512))
    hash_password = db.Column(db.String(512))

    def __init__(self, first_name, last_name, username, email, hash_password):
        self.first_name = first_name
        self.last_name = last_name
        self.username = username
        self.email = email
        self.hash_password = hash_password
