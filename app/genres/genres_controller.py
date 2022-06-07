from flask import request, current_app
from sqlalchemy.exc import SQLAlchemyError
from app import db
from app.movies.movieModel import Movie
from app.genres.genreModel import Genre

def create_genre():
    name = request.json.get("name", None)

    if not name:
        return {"msg": "Wrong name"}, 401

    genre = Genre(name=name)

    try:
        db.session.add(genre)
        db.session.commit()
    except SQLAlchemyError as e:
        current_app.logger.error(e)
        db.session.rollback()
    return {'id': genre.id, "name": genre.name}

def getGenresList():
    allGenres = db.session.query(Genre).all()
    for i in range(len(allGenres)):
        genre = allGenres[i].as_dict()
        allGenres[i] = genre
    return temp(allGenres)
    #allMoviesJson = json.dumps(allMovies, default=str)
    moviesDict = movieListToDict(allMovies)
    return moviesDict

def getGenre(genre_id):
    genre = Genre.query.filter(Genre.id == genre_id).first()
    genre = genre.as_dict()
    return genre

def movieListToDict(movies):
    movieDict = {}
    for i in range(len(movies)):
        key = "movie" + str(movies[i]['id'])
        movieDict[key] = movies[i]
    return movieDict

def temp(genres):
    genresDict = {}
    genresDict['genres'] = genres
    return genresDict