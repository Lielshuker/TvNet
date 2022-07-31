from app.movies import movies
from app.movies.movies_controller import create_movie, getMoviesList, getMovie

movies.route('/', methods=["POST"])(create_movie)
movies.route('/allMovies/<username>', methods=["GET"])(getMoviesList)
movies.route('/<movie_id>', methods=["GET"])(getMovie)