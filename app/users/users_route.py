from app.users import users
from app.users.user_controller import create_user

users.route('/', methods=["POST"])(create_user)
# users.route('/<user_id>', methods=["DELETE"])(delete_user)

