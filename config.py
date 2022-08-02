import os


class Config:
    FLASK_PORT = 5000
    FLASK_CONFIG = "development"
    DEBUG = True
    TESTING = False
    MAIL_SUPPRESS_SEND = False
    MAIL_DEBUG = True
    FRONT_END_URL = 'http://localhost:3000'
    SQLALCHEMY_DATABASE_URI = "mysql://root:1234@localhost:3306/TvNet"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.urandom(12)
    # os.environ['FLASK_RUN_FROM_CLI'] = 'false'


class ProdConfig(Config):

    # SQLALCHEMY_DATABASE_URI = "postgresql+psycopg2://collins:11946@localhost/watchlist_test"
    # todo
    pass


class TestConfig(Config):
    # SQLALCHEMY_DATABASE_URI = "postgresql+psycopg2://collins:11946@localhost/watchlist_test"
    # todo
    TESTING = False


class DevConfig(Config):
    SQLALCHEMY_DATABASE_URI = "mysql://root:1234@localhost:3306/TvNet"
    # todo
    DEBUG = True
    MAIL_DEBUG = True


config = {
    "development": DevConfig,
    "production": ProdConfig,
    "test": TestConfig
}