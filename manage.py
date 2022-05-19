from socket import SocketIO

from app import create_app

app = create_app("development")

if __name__ == "__main__":
    # socketio = SocketIO()
    flask_port = app.config['FLASK_PORT']
    # todo ssl_context='adhoc'
    app.run(app, debug=True, host='0.0.0.0', port=flask_port, threaded=True)
