from app.recommend_model import recommender
from app.recommend_model.recommender_controller import predict, update_rate, get_rate

recommender.route('/predict', methods=["GET"])(predict)
recommender.route('/rate/<user_id>', methods=["POST"])(update_rate)
recommender.route('/rate/<user_id>', methods=["GET"])(get_rate)

