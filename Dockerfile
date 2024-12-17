# Python 3.11 버전 이미지 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 종속성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Flask 애플리케이션 복사
COPY . .

# 환경 변수 설정 (Flask 실행용)
ENV FLASK_APP=index.py
ENV FLASK_RUN_HOST=0.0.0.0

# 포트 노출
EXPOSE 8080

# Flask 서버 실행
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]