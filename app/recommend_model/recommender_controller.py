import re

from flask import request, jsonify
import tensorflow as tf

from app.movies.movieModel import Movie
from app.recommend_model import retrieval_model
from app.users.UserModel import User
from app.watched_movies.watchMoviesModel import WatchedMovie
from app import db
import requests, json



MIN_CANDID = 100
NUM_RECOMMEND = 10


def predict():
    user_id = request.args.get("uid")
    user = User.query.filter(user_id).first()
    history = [m.movieid for m in user.movies]

    u_feature = {'userid': tf.constant([user.id])}
    candids = retrieval_model.predict(u_feature, num_candids=MIN_CANDID + len(history))

    candids = list(set(candids) - set(history))
    features = {'userid': tf.constant([user.userid] * len(candids)),
                'movieid': tf.constant(candids)
                }

    preds = ranking_model.predict(features)
    item_pred = sorted(list(zip(candids, preds)), key=lambda tup: -tup[1])[:NUM_RECOMMEND]
    out = {'recs': [t[0] for t in item_pred],
           'preds': [str(t[1]) for t in item_pred]
           }

    return jsonify(out)


def update_rate(user_id):
    movie_id = request.json.get("movie_id")
    rating = float(request.json.get("rating"))

    user = User.query.filter(User.user_id == user_id).first()
    if not user:
        return {"msg": "user not exist"}, 401  # todo error
    movie = Movie.query.filter(Movie.id == movie_id).first()
    if not movie:
        return {"msg": "movie not exist"}, 401  # todo error

    watched = WatchedMovie.query.filter(WatchedMovie.user_id == user_id).first()
    if not watched.movie_id:
        watched.rating = -1
        return

    watched.rating = rating
    db.session.add(watched)
    db.session.commit()


def get_recommendations(user_id):
    result = json.loads(requests.get('http://127.0.0.1:8000/v1/predict', params={'user_id': user_id}).content)
    # recommended_movies = [{'movie':Movie.query.get(m), 'pred': p} for m, p in zip(result['recs'], result['preds'])]
    recommended_movies = []
    for m in result['recs']:
        if (len(recommended_movies) == 10):
            break
        m = re.sub(' \(.*', '', m, flags=re.DOTALL)
        movie = Movie.query.filter(m == Movie.name).first()
        if movie:
            recommended_movies.append({'movie': movie})

    # recommended_movies = [{'movie':Movie.query.get(m)} for m in zip(result['recs'])]
    return recommended_movies


def get_rate(user_id):
    movie_id = request.json.get("movie_id")
    user = User.query.filter(User.user_id == user_id).first()
    if not user:
        return {"msg": "user not exist"}, 401  # todo error
    movie = Movie.query.filter(Movie.id == movie_id).first()
    if not movie:
        return {"msg": "movie not exist"}, 401  # todo error

    watched = WatchedMovie.query.filter(WatchedMovie.user_id == user_id).first()
    return {'rate': watched.rating}

