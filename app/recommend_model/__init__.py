from flask import Blueprint
recommender = Blueprint('recommender', __name__)
from . import recommender_route#, forms, error