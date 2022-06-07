from app import db

class MoviesGenres(db.Model):
    __tablename__ = 'movies_genres'
    movie_id = db.Column(db.Integer, primary_key=True)
    genre_id = db.Column(db.Integer, primary_key=True)

    def __init__(self, movie_id, genre_id):
        self.movie_id = movie_id
        self.genre_id = genre_id

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}