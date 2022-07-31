import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# Ratings data.
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train")

for x in ratings.take(1).as_numpy_iterator():
  pprint.pprint(x)

for x in movies.take(1).as_numpy_iterator():
  pprint.pprint(x)

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
})
movies = movies.map(lambda x: x["movie_title"])
# print(ratings.take(1)['user_id'].as_numpy_iterator())
# print(ratings['user_id'])


tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)


movie_titles = movies.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

unique_movie_titles[:10]

embedding_dimension = 32


user_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_user_ids, mask_token=None),
  # We add an additional embedding to account for unknown tokens.
  tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
])

movie_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_movie_titles, mask_token=None),
  tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
])

metrics = tfrs.metrics.FactorizedTopK(
  candidates=movies.batch(128).map(movie_model)
)

task = tfrs.tasks.Retrieval(
  metrics=metrics
)


class MovielensModel(tfrs.Model):

  def __init__(self, user_model, movie_model):
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

class NoBaseClassMovielensModel(tf.keras.Model):

  def __init__(self, user_model, movie_model):
    super().__init__()
    self.movie_model: tf.keras.Model = movie_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def train_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

    # Set up a gradient tape to record gradients.
    with tf.GradientTape() as tape:

      # Loss computation.
      user_embeddings = self.user_model(features["user_id"])
      positive_movie_embeddings = self.movie_model(features["movie_title"])
      loss = self.task(user_embeddings, positive_movie_embeddings)

      # Handle regularization losses as well.
      regularization_loss = sum(self.losses)

      total_loss = loss + regularization_loss

    gradients = tape.gradient(total_loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics["loss"] = loss
    metrics["regularization_loss"] = regularization_loss
    metrics["total_loss"] = total_loss

    return metrics

  def test_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

    # Loss computation.
    user_embeddings = self.user_model(features["user_id"])
    positive_movie_embeddings = self.movie_model(features["movie_title"])
    loss = self.task(user_embeddings, positive_movie_embeddings)

    # Handle regularization losses as well.
    regularization_loss = sum(self.losses)

    total_loss = loss + regularization_loss

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics["loss"] = loss
    metrics["regularization_loss"] = regularization_loss
    metrics["total_loss"] = total_loss

    return metrics

model = MovielensModel(user_model, movie_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

model.fit(cached_train, epochs=1)
model.evaluate(cached_test, return_dict=True)
#

# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
# recommends movies out of the entire movies dataset.
index.index_from_dataset(
  tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
)

# Get recommendations.
_, titles = index(tf.constant(["42"]))
print(f"Recommendations for user 42: {titles[0, :3]}")


# Export the query model.
# with tempfile.TemporaryDirectory() as tmp:
# path = os.path.join('./recommendation_model', "model")

# Save the index.
# tf.saved_model.save(index, './recommendation_model/model')
model.user_model.save('./recommendation_model/user_model')
model.movie_model.save('./recommendation_model/movie_model')

# Load it back; can also be done in TensorFlow Serving.
# loaded = tf.saved_model.load('./recommendation_model/model')
#
# # Pass a user id in, get top predicted movie titles back.
# scores, titles = loaded(["42"])
#
# print(f"Recommendations: {titles[0][:3]}")


new_user_ID = 0

# The format of each line is (userID, movieID, rating)
new_user_ratings = [
     (0,260,9), # Star Wars (1977)
     (0,1,8), # Toy Story (1995)
     (0,16,7), # Casino (1995)
     (0,25,8), # Leaving Las Vegas (1995)
     (0,32,9), # Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
     (0,335,4), # Flintstones, The (1994)
     (0,379,3), # Timecop (1994)
     (0,296,7), # Pulp Fiction (1994)
     (0,858,10) , # Godfather, The (1972)
     (0,50,8) # Usual Suspects, The (1995)
    ]

ratings.concatenate(new_user_ratings)
# new_user_ratings_RDD = sc.parallelize(new_user_ratings)
# print 'New user ratings: %s' % new_user_ratings_RDD.take(10)









# scann_index = tfrs.layers.factorized_top_k.ScaNN(model.user_model)
# scann_index.index_from_dataset(
#   tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
# )
# # Get recommendations.
# _, titles = scann_index(tf.constant(["42"]))
# print(f"Recommendations for user 42: {titles[0, :3]}")
#
# # Export the query model.
# with tempfile.TemporaryDirectory() as tmp:
#   path = os.path.join(tmp, "model")
#
#   # Save the index.
#   tf.saved_model.save(
#       index,
#       path,
#       options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"])
#   )
#
#   # Load it back; can also be done in TensorFlow Serving.
#   loaded = tf.saved_model.load(path)
#
#   # Pass a user id in, get top predicted movie titles back.
#   scores, titles = loaded(["42"])
#
#   print(f"Recommendations: {titles[0][:3]}")


# ratings = tfds.load("movielens/100k-ratings", split="train")
#
# ratings = ratings.map(lambda x: {
#     "movie_title": x["movie_title"],
#     "user_id": x["user_id"],
#     "user_rating": x["user_rating"]
# })
#
# tf.random.set_seed(42)
# shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
#
# train = shuffled.take(80_000)
# test = shuffled.skip(80_000).take(20_000)
#
# movie_titles = ratings.batch(1_000_000).map(lambda x: x["movie_title"])
# user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])
#
# unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
# unique_user_ids = np.unique(np.concatenate(list(user_ids)))
#
#
# class RankingModel(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         embedding_dimension = 32
#
#         # Compute embeddings for users.
#         self.user_embeddings = tf.keras.Sequential([
#           tf.keras.layers.StringLookup(
#             vocabulary=unique_user_ids, mask_token=None),
#           tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
#         ])
#
#         # Compute embeddings for movies.
#         self.movie_embeddings = tf.keras.Sequential([
#           tf.keras.layers.StringLookup(
#             vocabulary=unique_movie_titles, mask_token=None),
#           tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
#         ])
#
#         # Compute predictions.
#         self.ratings = tf.keras.Sequential([
#           # Learn multiple dense layers.
#           tf.keras.layers.Dense(256, activation="relu"),
#           tf.keras.layers.Dense(64, activation="relu"),
#           # Make rating predictions in the final layer.
#           tf.keras.layers.Dense(1)
#       ])
#
#     def call(self, inputs):
#
#         user_id, movie_title = inputs
#
#         user_embedding = self.user_embeddings(user_id)
#         movie_embedding = self.movie_embeddings(movie_title)
#
#         return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))
#
#
# RankingModel()((["42"], ["One Flew Over the Cuckoo's Nest (1975)"]))
# task = tfrs.tasks.Ranking(
#   loss = tf.keras.losses.MeanSquaredError(),
#   metrics=[tf.keras.metrics.RootMeanSquaredError()]
# )
#
#
# class MovielensModel(tfrs.models.Model):
#
#     def __init__(self):
#         super().__init__()
#         self.ranking_model: tf.keras.Model = RankingModel()
#         self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
#           loss = tf.keras.losses.MeanSquaredError(),
#           metrics=[tf.keras.metrics.RootMeanSquaredError()]
#         )
#
#     def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
#         return self.ranking_model((features["user_id"], features["movie_title"]))
#
#   def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
#     labels = features.pop("user_rating")
#
#     rating_predictions = self(features)
#
#     # The task computes the loss and the metrics.
#     return self.task(labels=labels, predictions=rating_predictions)
#
#
# model = MovielensModel()
# model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
# cached_train = train.shuffle(100_000).batch(8192).cache()
# cached_test = test.batch(4096).cache()
# model.fit(cached_train, epochs=3)
# model.evaluate(cached_test, return_dict=True)
#
