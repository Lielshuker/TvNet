from socket import SocketIO
from app import create_app, mail

app = create_app("development")

if __name__ == "manage":
    # socketio = SocketIO()
    flask_port = app.config['FLASK_PORT']
    # todo ssl_context='adhoc'
    app.run(app, debug=True, port=flask_port, threaded=True)