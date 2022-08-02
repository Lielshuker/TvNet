from app import db

class Movie(db.Model):
    __tablename__ = 'movies'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(512))
    length = db.Column(db.Integer)
    #genre_id = db.Column(db.Integer)
    genre_id = []
    release_year = db.Column(db.Date)
    date_added = db.Column(db.Date)
    description = db.Column(db.Text)
    movie_url = db.Column(db.String(512))
    image_url = db.Column(db.String(512))
    movielens_id = db.Column(db.Integer)

    def __int__(self, name, length, genre_id, release_year, date_added,
                description, movie_url, image_url, movielens_id):
    #def __int__(self, name, length, release_year, date_added,
    #            description, movie_url, image_url):
        self.name = name
        self.length = length
        self.genre_id = genre_id
        self.release_year = release_year
        self.date_added = date_added
        self.description = description
        self.movie_url = movie_url
        self.image_url = image_url
        self.movielens_id = movielens_id

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}