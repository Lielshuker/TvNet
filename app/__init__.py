# import os
#
from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy

# from app.auth import auth
from config import config
from flask_cors import CORS
from flask_jwt_extended import create_access_token,get_jwt,get_jwt_identity, \
                               unset_jwt_cookies, jwt_required, JWTManager

db = SQLAlchemy()

def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    # # conf.init_app(app)
    db.init_app(app)
    cors = CORS(app)
    app.config["JWT_SECRET_KEY"] = "secret-tvnet"
    jwt = JWTManager(app)
    # from .main import main as main_blueprint
    # app.register_blueprint(main_blueprint)
    #
    # from .auth import auth as auth_blueprint
    # app.register_blueprint(auth_blueprint)

    # from .api import api as api_blueprint
    # app.register_blueprint(api_blueprint, url_prefix='/api')
    app = setup_standard_api_gateway(app)
    # from .users import UserModel
    # with app.app_context():
    #     db.create_all()
    # db.create_all()
    # db.session.commit()

    # app.config['CORS_HEADERS'] = 'Content-Type'

    return app


def setup_standard_api_gateway(app):
    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    from .auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint, url_prefix='/auth')

    from .users import users as users_blueprint
    app.register_blueprint(users_blueprint, url_prefix='/users')
    return app