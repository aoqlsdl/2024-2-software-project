# 2024-2-software-project
2024-1 융합소프트웨어프로젝트 기말 과제(백엔드)

## 📋 프로젝트 개요

이 프로젝트는 **Flask**를 기반으로 장애학생과 수업 데이터를 매칭하는 시스템입니다.  
프론트엔드와 http 통신을 통해 사용자의 요청 데이터를 인코딩하고 머신러닝 모델을 활용하여 가장 유사한 결과를 예측하는 기능을 제공합니다.

---

## ⚙️ 주요 기능

1. **모델 기반 예측**  
   - 미리 학습된 머신러닝 모델을 통해 매칭 확률을 계산
   - 입력 데이터를 기반으로 가장 적합한 수업을 찾아 제공

2. **데이터 전처리**  
   - **OrdinalEncoder**를 사용하여 입력 및 증강 데이터를 수치화  
   - 요일/시간 등의 복잡한 데이터를 병합

3. **유사도 분석**  
   - **Cosine Similarity**를 사용해 요청 데이터와 증강 데이터의 유사도를 계산

4. **API 엔드포인트 제공**  
   - POST 요청으로 JSON 데이터를 입력받고 매칭 결과를 반환

---

## 🛠️ 설치 및 실행 방법

✅ 파이썬 버전은 **3.10.0 이하**로 설정해주셔야 합니다.

### 1. 가상 환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # MacOS/Linux
venv\Scripts\activate     # Windows
```

### 2. 필수 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. flask 실행
```bash
flask run
```
