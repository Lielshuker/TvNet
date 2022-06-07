import json
from datetime import datetime, timezone, timedelta
from flask import request, jsonify, render_template
from flask_jwt_extended import create_access_token, unset_jwt_cookies, get_jwt, get_jwt_identity, jwt_required
from itsdangerous import URLSafeTimedSerializer
from passlib.handlers.pbkdf2 import pbkdf2_sha256
from flask import current_app
from app import db
from app.users.EmailModel import create_message, service, send_message
from app.users.UserModel import User
from threading import Thread


@jwt_required()
def profile():
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


def forgot_password():
    email = request.args.get("email", None)
    user = User.query.filter(User.email == email).first()
    if not user:
        return {"msg": "Invalid email address!"}, 401  # todo error
    send_password_reset_link(user.email)
    return {"message": 200}


def send_password_reset_link(email):
    password_reset_serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
    base_url = current_app.config['FRONT_END_URL'] + '/reset/'
    base_url += password_reset_serializer.dumps(email, salt='password-reset-salt')
    # password_reset_url = url_for(
    #     'auth.token_reset',
    #     token=password_reset_serializer.dumps(email, salt='password-reset-salt'),
    #     _external=True)
    html = render_template('email_reset.html',
        password_reset_url=base_url)

    send_email('Password Reset Requested', email, html)


def send_email(subject, recipients, html_body):
    message = create_message('TvNet', recipients, subject,  html_body)
    thr = Thread(target=send_async_email, args=[message])
    thr.start()


def send_async_email(msg):
    try:
        send_message(service=service, user_id='me', message=msg)
    except Exception as e:
        print(e)


def token_reset(token):
    try:
        password = request.args.get("password", None)
        password_reset_serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
        email = password_reset_serializer.loads(token, salt='password-reset-salt', max_age=3600)
    except:
        return {"msg": "The password reset link is invalid or has expired"}, 401  # todo error

    user = User.query.filter(User.email == email).first()
    if not user:
        return {"msg": "Invalid email address!"}, 401  # todo error

    user.hash_password = pbkdf2_sha256.hash(password)
    db.session.add(user)
    db.session.commit()
    return {"message": 200}


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