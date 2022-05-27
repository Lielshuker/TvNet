from flask import request, current_app
from passlib.hash import pbkdf2_sha256
from sqlalchemy.exc import SQLAlchemyError

from app import db
# from app.errors import InvalidAPIUsage
from app.users.UserModel import User


def create_user():
    first_name = request.json.get("first_name", None)
    last_name = request.json.get("last_name", None)
    email = request.json.get("email", None)
    username = request.json.get("username", None)
    password = request.json.get("password", None)

    # if not first_name:
    #     raise InvalidAPIUsage("No first name provided!")
    # if not last_name:
    #     raise InvalidAPIUsage("No last name provided!")
    # if not email:
    #     raise InvalidAPIUsage("No email provided!")
    # if not username:
    #     raise InvalidAPIUsage("No username provided!")
    # if not password:
    #     raise InvalidAPIUsage("No password provided!")

    user = User.query.filter(User.email == email).first()
    if user:
        return {"msg": "user already exist"}, 401  # todo error
    hash_password = pbkdf2_sha256.hash(password)
    user = User(first_name=first_name, last_name=last_name, username=username, email=email, hash_password=hash_password)

    try:
        db.session.add(user)
        db.session.commit()
    except SQLAlchemyError as e:
        current_app.logger.error(e)
        db.session.rollback()
        # return pretty_result(code.DB_ERROR) todo
    return {'id': user.id, 'first_name': first_name, 'last_name': last_name, 'username': username, 'email': email}
