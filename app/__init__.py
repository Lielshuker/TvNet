from flask import Flask
from flask_mail import Mail
from flask_sqlalchemy import SQLAlchemy
from config import config
from flask_cors import CORS
from flask_jwt_extended import JWTManager

db = SQLAlchemy()
mail = Mail()


def create_app(config_name):
    app = Flask(__name__, template_folder='templates')
    app.config.from_object(config[config_name])
    # # conf.init_app(app)
    db.init_app(app)
    cors = CORS(app)
    app.config["JWT_SECRET_KEY"] = "secret-tvnet"
    jwt = JWTManager(app)
    app = setup_standard_api_gateway(app)
    mail.init_app(app)
    return app


def setup_standard_api_gateway(app):
    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    from .auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint, url_prefix='/auth')

    from .users import users as users_blueprint
    app.register_blueprint(users_blueprint, url_prefix='/users')

    from .movies import movies as movies_blueprint
    app.register_blueprint(movies_blueprint, url_prefix='/movies')
    return app