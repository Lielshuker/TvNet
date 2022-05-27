from . import auth
from .auth_controller import login, logout, refresh_expiring_jwts, profile

auth.route('/login', methods=["POST"])(login)
auth.route("/logout", methods=["POST"])(logout)
# auth.after_request(refresh_expiring_jwts)
auth.route('/profile', methods=["GET"])(profile)
