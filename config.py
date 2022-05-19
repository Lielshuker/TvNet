class Config:
    FLASK_PORT = 5000
    FLASK_CONFIG = "development"
    DEBUG = False
    TESTING = False
    SQLALCHEMY_DATABASE_URI = "mysql://root:1234@localhost:3306/TvNet"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # @staticmethod
    # def init_app(app):
    #     Config.the_app = app


class ProdConfig(Config):

    # SQLALCHEMY_DATABASE_URI = "postgresql+psycopg2://collins:11946@localhost/watchlist_test"
    # todo
    pass


class TestConfig(Config):
    # SQLALCHEMY_DATABASE_URI = "postgresql+psycopg2://collins:11946@localhost/watchlist_test"
    # todo
    TESTING = True


class DevConfig(Config):
    # MYSQL_HOST = 'localhost'
    # MYSQL_USER = 'root'
    # MYSQL_PASSWORD = '1234'
    # MYSQL_PORT = 3306
    # MYSQL_DB =
    SQLALCHEMY_DATABASE_URI = "mysql://root:1234@localhost:3306/TvNet"
    # todo
    DEBUG = True


config = {
    "development": DevConfig,
    "production": ProdConfig,
    "test": TestConfig
}