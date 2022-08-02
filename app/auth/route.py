from . import auth
from .auth_controller import login, logout, refresh_expiring_jwts, forgot_password, token_reset

auth.route('/login', methods=["POST"])(login)
auth.route("/logout", methods=["POST"])(logout)
# auth.after_request(refresh_expiring_jwts)
auth.route('/forgot_password', methods=["GET"])(forgot_password)
auth.route('/reset/<token>', methods=["GET", "POST"])(token_reset)


