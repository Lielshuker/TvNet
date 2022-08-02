import json
from flask import request, current_app
from sqlalchemy.exc import SQLAlchemyError
from app import db
from app.movies.movieModel import Movie
from app.genres.genreModel import Genre
from app.movies_genres.MoviesGenresModel import MoviesGenres
from datetime import datetime

from app.recommend_model.recommender_controller import get_recommendations
from app.users.UserModel import User
from app.watched_movies.watchMoviesModel import WatchedMovie


def create_movie():
    name = request.json.get("name", None)
    length = request.json.get("length", None)
    genre_id = request.json.get("genre_id", None)
    release_year = request.json.get("release_year", None)
    date_added = request.json.get("date_added", None)
    description = request.json.get("description", None)
    movie_url = request.json.get("movie_url", None)
    image_url = request.json.get("image_url", None)

    if not name:
        return {"msg": "Wrong name"}, 401
    if not length:
        return {"msg": "Wrong length"}, 401
    if not genre_id:
        return {"msg": "Wrong genre_id"}, 401
    if not release_year:
        return {"msg": "Wrong release_year"}, 401
    if not date_added:
        return {"msg": "Wrong date_added"}, 401
    if not description:
        return {"msg": "Wrong description"}, 401
    if not movie_url:
        return {"msg": "Wrong movie_url"}, 401
    if not image_url:
        return {"msg": "Wrong image_url"}, 401

    # Convert strings to datetime objects
    release_year = datetime.strptime(str(release_year), '%Y')
    release_year = release_year.year

    date_added = datetime.strptime(date_added, '%d.%m.%Y')
    date_added = date_added.date()

    #movie = Movie(name=name, length=length, genre_id=genre_id, release_year=release_year, date_added=date_added,
    #              description=description, movie_url=movie_url, image_url=image_url)

    movie = Movie(name=name, length=length, release_year=release_year, date_added=date_added, description=description,
                  movie_url=movie_url, image_url=image_url)

    try:
        db.session.add(movie)
        db.session.commit()
    except SQLAlchemyError as e:
        current_app.logger.error(e)
        db.session.rollback()
    #return {'id': movie.id, "name": movie.name, "length": movie.length, "release_year": movie.release_year,
    #        "date_added": movie.date_added, "description": movie.description, "movie_url": movie.movie_url,
    #        "image_url": movie.image_url }
    return {'id': movie.id, "name": movie.name, "length": movie.length, "genre_id": movie.genre_id,
            "release_year": movie.release_year, "date_added": movie.date_added, "description": movie.description,
            "movie_url": movie.movie_url, "image_url": movie.image_url}


def getMoviesList(username):

    page = request.json.get("page", 0)
    per_page = request.json.get("per_page", 10)
    user = User.query.filter(User.username == username).first()
    if not user:
        return {"msg": "user not exist"}, 401  # todo error
    watch_movie = WatchedMovie.query.filter(WatchedMovie.user_id == user.id).all()
    if len(watch_movie) > 10:
        allMovies = []
        movies = get_recommendations(user_id=user.id)
        total = 10
        for movie in movies:
            # movie_db = Movie.query.filter(Movie.name == movie).first()
            # allMovies.append(movie_db)
            movie_db = movie['movie']
            if movie_db:
                allMovies.append(movie_db)


    else:
        allMovies = db.session.query(Movie).paginate(page,per_page,error_out=False)
        total  = allMovies.total
        allMovies = allMovies.items
    for i in range(len(allMovies)):
        movie = allMovies[i].as_dict()
        movie['date_added'] = dateToString(movie['date_added'])
        genres = getGenresList(movie['id'])
        movie['genre_id'] = genres
        allMovies[i] = movie

    return temp(allMovies, total)
    #allMoviesJson = json.dumps(allMovies, default=str)
    moviesDict = movieListToDict(allMovies)
    return moviesDict

def getMovie(movie_id):
    movie = Movie.query.filter(Movie.id == movie_id).first()
    movie = movie.as_dict()
    movie['date_added'] = dateToString(movie['date_added'])
    genres = getGenresList(movie['id'])
    movie['genre_id'] = genres
    #movieJson = json.dumps(movie.as_dict(), default=str)
    return movie

def dateToString(date):
    dateStr = date.strftime("%Y - %m - %d")
    return dateStr

def movieListToDict(movies):
    movieDict = {}
    for i in range(len(movies)):
        key = "movie" + str(movies[i]['id'])
        movieDict[key] = movies[i]
    return movieDict

def temp(movies, total):
    movieDict = {}
    movieDict['movies'] = movies
    movieDict['total'] = total
    return movieDict

def getGenresList(movie_id):
    genres = MoviesGenres.query.filter(MoviesGenres.movie_id == movie_id).all()
    # To list
    for i in range(len(genres)):
        genreID = genres[i].genre_id
        genre = Genre.query.filter(Genre.id == genreID).first()
        genres[i] = genre.name
    return genres