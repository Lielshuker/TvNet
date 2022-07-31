from app import db


class WatchedMovie(db.Model):
    __tablename__ = 'watched_movies'
    id = db.Column(db.Integer, primary_key=True)
    movie_id = db.Column(db.Integer)
    user_id = db.Column(db.Integer)
    watch_num = db.Column(db.Integer)
    date_modify = db.Column(db.DATE)

    def __init__(self, movie_id, user_id, watch_num, date_modify):
        self.movie_id = movie_id
        self.user_id = user_id
        self.watch_num = watch_num
        self.date_modify = date_modify

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}