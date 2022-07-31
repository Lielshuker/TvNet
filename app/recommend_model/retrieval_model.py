import importlib
import datetime
import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

embedding_dimension = 32


def load_data(ratings, movies):


    # Let's use a random split, putting 80% of the ratings in the train set, and 20% in the test set.
    tf.random.set_seed(42)
    shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(80_000)
    test = shuffled.skip(80_000).take(20_000)

    # Next we figure out unique user ids and movie titles present in the data so that
    # we can create the embedding user and movie embedding tables.
    movie_titles = movies.batch(1_000)
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))
    return unique_user_ids, unique_movie_titles, train, test


class MovielensModel(tfrs.Model):

  def __init__(self, user_model, movie_model, task):
    super().__init__()
    self.movie_model: tf.keras.Model = movie_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["user_id"])
    # And pick out the movie features and pass them into the movie model,
    # getting embeddings back.
    positive_movie_embeddings = self.movie_model(features["movie_title"])

    # The task computes the loss and the metrics.
    return self.task(user_embeddings, positive_movie_embeddings)


def creating_model_config(unique_user_ids, unique_movie_titles, movies):
    # The first step is to decide on the dimensionality of the query and candidate representations:
    embedding_dimension = 32

    # The second is to define the model itself.
    user_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
        # We add an additional embedding to account for unknown tokens.
        tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    # We can do the same with the candidate tower.
    movie_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_movie_titles, mask_token=None),
        tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
    ])

    # We use the tfrs.metrics.FactorizedTopK metric for our retrieval model.
    metrics = tfrs.metrics.FactorizedTopK(
        candidates=movies.batch(128).map(movie_model)
    )

    # The next component is the loss used to train our model. We'll make use of the Retrieval task object:
    # a convenience wrapper that bundles together the loss function and metric computation:
    task = tfrs.tasks.Retrieval(
        metrics=metrics
    )
    return user_model, movie_model, task


def build_model(ratings, movies):

    unique_user_ids, unique_movie_titles, train, test = load_data(ratings, movies)
    user_model, movie_model, task = creating_model_config(unique_user_ids, unique_movie_titles, movies)
    model = MovielensModel(user_model, movie_model, task)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    cached_train = train.shuffle(100_000).batch(8192).cache()
    cached_test = test.batch(4096).cache()

    model.fit(cached_train, epochs=3)
    model.evaluate(cached_test, return_dict=True)
    index = prediction(model, movies)
    # index(tf.constant(["42"]))
    # tf.keras.models.save_model(model=model, filepath='./recommendation_model/model')
    save_model(model, index)


def run():
    # Ratings data.
    ratings = tfds.load("movielens/100k-ratings", split="train")
    # Features of all the available movies.
    movies = tfds.load("movielens/100k-movies", split="train")
    # Keep only 'movie_title' and 'user_id'
    ratings = ratings.map(lambda x: {
        "movie_title": x["movie_title"],
        "user_id": x["user_id"],
    })
    movies = movies.map(lambda x: x["movie_title"])
    build_model(ratings, movies)
    # dict_ = {  # 'bucketized_user_age': [45.0],
    #     'movie_title': [b"One Flew Over the Cuckoo's Nest (1975)"],
    #     'user_id': [b'0']
    # }
    # dataset = tf.data.Dataset.from_tensor_slices(dict_)
    # for features in dataset:
    #     print(features['user_id'])
    #
    # ratings.concatenate(dataset)
    # build_model(ratings, movies)



def prediction(model, movies):
    # Create a model that takes in raw query features, and
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    # recommends movies out of the entire movies dataset.
    index.index_from_dataset(
        tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
    )

    # Get recommendations.
    _, titles = index(tf.constant(["0"]))
    print(f"Recommendations for user 42: {titles[0, :3]}")
    return index

def predict(model, features, num_candids = 100):
    query_embedding = model.predict(features).squeeze()


def save_model(model, index):
    model.save_weights('./recommendation_model/model')
    tf.saved_model.save(
        index,
        './recommendation_model/index',
        options=tf.saved_model.SaveOptions(namespace_whitelist=["Index"])
    )    # model.user_model.save_weights('./recommendation_model/user_model')
    # model.movie_model.save_weights('./recommendation_model/movie_model')


def load_model(model):
    model = model.load_weights('./recommendation_model/model')
    index = tf.saved_model.load('./recommendation_model/index')

    # user_model = model.user_model.load_weights('./recommendation_model/user_model')
    # movie_model = model.movie_model.load_weights('./recommendation_model/movie_model')
    # task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
    #     movies.batch(128).map(movie_model)))
    # model = MovielensModel(user_model, movie_model, task)

    return model, index

# run()
#
# ratings = tfds.load("movielens/100k-ratings", split="train")
# # Features of all the available movies.
# movies = tfds.load("movielens/100k-movies", split="train")
# # Keep only 'movie_title' and 'user_id'
# ratings = ratings.map(lambda x: {
#     "movie_title": x["movie_title"],
#     "user_id": x["user_id"],
# })
# movies = movies.map(lambda x: x["movie_title"])
#
# unique_user_ids, unique_movie_titles, train, test = load_data(ratings, movies)
# user_model, movie_model, task = creating_model_config(unique_user_ids, unique_movie_titles, movies)
# model = MovielensModel(user_model, movie_model, task)
#
# model, index = load_model(model)
# model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
#
# cached_train = train.shuffle(100_000).batch(8192).cache()
# cached_test = test.batch(4096).cache()
# model.fit(cached_train, epochs=3)
#
# # # Use brute-force search to set up retrieval using the trained representations.
# # index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
# # index.index(movies.batch(100).map(model.movie_model), movies)
#
# # Get some recommendations.
# _, titles = index(np.array(["42"]))
# print(f"Top 3 recommendations for user 42: {titles[0, :3]}")
# # scores, titles = model.user_model.predict(['None'])
# #
# #
# # print(f"Recommendations: {titles[0][:5]}")
#



