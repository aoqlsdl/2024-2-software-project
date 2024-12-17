# import os
from flask import Flask
# from flask_sqlalchemy import SQLAlchemy
# from flask_migrate import Migrate
from flask_cors import CORS

# 중앙에서 확장 생성
# db = SQLAlchemy()
# migrate = Migrate()

def create_app():
    app = Flask(__name__)

    # app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'mysql+pymysql://myuser:mypassword@localhost/softwareproject')
    # app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # # 확장 초기화
    # db.init_app(app)
    # migrate.init_app(app, db)
    CORS(app)

    # Blueprint 등록
    from .match.match_controller import match_bp
    app.register_blueprint(match_bp, url_prefix='/match')

    # 에러 핸들러 등록
    from .match.exception import CustomException, handle_custom_exception
    app.register_error_handler(CustomException, handle_custom_exception)

    return app