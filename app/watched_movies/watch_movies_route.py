from app.watched_movies import watched_movies
from app.watched_movies.watch_movies_comtroller import add_watched_movie

watched_movies.route('/<movie_id>', methods=["POST"])(add_watched_movie)

