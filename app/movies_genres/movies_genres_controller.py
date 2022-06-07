from flask import request, current_app
from sqlalchemy.exc import SQLAlchemyError
from app import db
from app.movies_genres.MoviesGenresModel import MoviesGenres


def add_genre_to_movie():
    movie_id = request.json.get("movie_id", None)
    genre_id = request.json.get("genre_id", None)

    if not movie_id:
        return {"msg": "Wrong movie_id"}, 401  # todo error
    if not genre_id:
        return {"msg": "Wrong genre_id"}, 401  # todo error

    relation = MoviesGenres(movie_id=movie_id, genre_id=genre_id)

    try:
        db.session.add(relation)
        db.session.commit()
    except SQLAlchemyError as e:
        current_app.logger.error(e)
        db.session.rollback()
        # return pretty_result(code.DB_ERROR) todo
    return {'movie_id': relation.movie_id, 'genre_id': relation.genre_id}
