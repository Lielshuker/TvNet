from flask import request, current_app
from sqlalchemy.exc import SQLAlchemyError

from app import db
from app.movies.movieModel import Movie
from app.users.UserModel import User
from app.watched_movies.watchMoviesModel import WatchedMovie
from datetime import datetime, timedelta


def add_watched_movie(movie_id):

    participants = request.json.get("participants", None)
    date = request.json.get("date", None)
    date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    movie = Movie.query.filter(Movie.id == movie_id).first()
    users_mails = []
    # for participant in participants:
    #     user = User.query.filter(User.username == participant).first()
    #     users_mails.append(user.email)
    if not movie:
        return {"msg": "Wrong movie"}, 401

    for user in participants:
        user_db = User.query.filter(user == User.username).first()
        watched_movie = WatchedMovie.query.filter(WatchedMovie.user_id == user_db.id).first()
        if not watched_movie:
            watched_movie = WatchedMovie(movie_id=movie_id, user_id=user_db.id, watch_num=1, date_modify=date)
        else:
            if watched_movie.date_modify + timedelta(days=1) < date:
                watched_movie.watch_num += 1
                watched_movie.date_modify = date
            else:
                continue
        try:
            db.session.add(watched_movie)
            db.session.commit()
        except SQLAlchemyError as e:
            current_app.logger.error(e)
            db.session.rollback()

    return {"message": 200}





