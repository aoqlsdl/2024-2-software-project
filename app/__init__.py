from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Blueprint 등록
    from .match.match_controller import match_bp
    app.register_blueprint(match_bp, url_prefix='/match')

    return app
