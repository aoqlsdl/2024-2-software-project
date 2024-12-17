from app import create_app
import pandas as pd

app = create_app()


# 매핑 출력(디버깅용)
with app.app_context():
    print(app.url_map)

# flask 앱 실행
if __name__ == '__main__':
    app.run(debug=True)