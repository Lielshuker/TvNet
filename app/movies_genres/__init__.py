from flask import Blueprint
users = Blueprint('movies_genres', __name__)
from . import movies_genres_route