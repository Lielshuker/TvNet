from flask import Blueprint
genres = Blueprint('genres', __name__)
from . import genres_route