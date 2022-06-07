from app.genres import genres
from app.genres.genres_controller import create_genre, getGenresList, getGenre

genres.route('/', methods=["POST"])(create_genre)
genres.route('/allGenres', methods=["GET"])(getGenresList)
genres.route('/<genre_id>', methods=["GET"])(getGenre)