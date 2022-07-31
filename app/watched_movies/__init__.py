from flask import Blueprint
watched_movies = Blueprint('watched_movies', __name__)
from . import watch_movies_route#, forms, error