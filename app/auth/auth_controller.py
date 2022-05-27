import json
from datetime import datetime, timezone, timedelta
from flask import request, jsonify

from flask_jwt_extended import create_access_token, unset_jwt_cookies, get_jwt, get_jwt_identity, jwt_required
from passlib.handlers.pbkdf2 import pbkdf2_sha256

from app.users.UserModel import User


@jwt_required()
def profile(user_id, request):
    response_body = {
        "name": "Nagato",
        "about": "Hello! I'm a full stack developer that loves python and javascript"
    }
    return response_body


def login():
    username = request.json.get("username", None)
    password = request.json.get("password", None)
    if username is None:
        return {"msg": "Wrong username"}, 401 # todo error
    if password is None:
        return {"msg": "Wrong password"}, 401 # todo error
    # to check login
    user = User.query.filter(User.username == username).first()
    if not user:
        return {"msg": "user not exist"}, 401  # todo error
    if not pbkdf2_sha256.verify(password, user.hash_password):
        return {"msg": "Wrong password"}, 401  # todo error
    # todo check t he user in db
    access_token = create_access_token(identity='username')
    response = {"access_token": access_token}
    return response


def logout():
    # todo logout
    response = jsonify({"msg": "logout successful"})
    unset_jwt_cookies(response)
    return response.json


# The generated token always has a lifespan after which it expires.
# To ensure that this does not happen while the user is logged in,
# you have to create a function that refreshes the token when it is close to the end of its lifespan.
def refresh_expiring_jwts(response):
    try:
        exp_timestamp = get_jwt()["exp"]
        now = datetime.now(timezone.utc)
        target_timestamp = datetime.timestamp(now + timedelta(minutes=30))
        if target_timestamp > exp_timestamp:
            access_token = create_access_token(identity=get_jwt_identity())
            data = response.get_json()
            if type(data) is dict:
                data["access_token"] = access_token
                response.data = json.dumps(data)
        return response
    except (RuntimeError, KeyError):
        # Case where there is not a valid JWT. Just return the original respone
        return response