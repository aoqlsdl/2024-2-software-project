from flask import Blueprint, request, jsonify
import os
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Flask Blueprint 설정
match_bp = Blueprint('match', __name__)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 전역 변수 설정
ordinal_encoders = {}
base_path = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(base_path, 'data', 'final_version', 'Ewha_final_model.pkl')
ctgan_data_path = os.path.join(base_path, 'data', 'Ewha_final.csv')

# 모델 로드
try:
    with open(model_path, 'rb') as file:
        model = joblib.load(file)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise


# 인코더 초기화
def initialize_encoders():
    global ordinal_encoders
    try:
        encoder_files = {
            'disType': 'ordinal_encoder_disType.pkl',
            'classDept': 'ordinal_encoder_classDept.pkl',
            'helpType': 'ordinal_encoder_helpType.pkl',
            'classDate': 'ordinal_encoder_classDate.pkl',
            'classLocation': 'ordinal_encoder_classLocation.pkl',
            'className': 'ordinal_encoder_className.pkl',
            'classTime': 'ordinal_encoder_classTime.pkl',
            'Credit': 'ordinal_encoder_Credit.pkl',
            'haksuNum': 'ordinal_encoder_haksuNum.pkl',
            'profName': 'ordinal_encoder_profName.pkl',
            'studentNum': 'ordinal_encoder_studentNum.pkl',
            'subjectArea': 'ordinal_encoder_subjectArea.pkl',
            'subjectArea2': 'ordinal_encoder_subjectArea2.pkl',
        }
        for key, file_name in encoder_files.items():
            ordinal_encoders[key] = joblib.load(os.path.join(base_path, 'data', 'final_version', file_name))
        logging.info("Encoders initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing encoders: {e}")
        raise


def merge_class_times(class_time_list):
    """
    시간 데이터를 병합하여 CSV 형식(월3~4)으로 변환
    """
    time_dict = defaultdict(list)
    for time_entry in class_time_list:
        time_id = time_entry.get("timeId")
        if time_id and len(time_id) > 1:
            day, hour = time_id[:1], int(time_id[1:])
            time_dict[day].append(hour)

    merged_times = []
    for day, hours in time_dict.items():
        hours = sorted(hours)
        start = hours[0]
        for i in range(1, len(hours)):
            if hours[i] != hours[i - 1] + 1:
                merged_times.append(f"{day}{start}~{hours[i - 1]}")
                start = hours[i]
        merged_times.append(f"{day}{start}~{hours[-1]}")
    return merged_times

def transform_column(column_name, column_data, encoder):
    """
    OrdinalEncoder로 변환 실패 시 기본값(-1) 반환
    """
    try:
        column_df = pd.DataFrame(column_data, columns=[column_name])
        return encoder.transform(column_df).flatten()
    except ValueError as e:
        print(f"ValueError in {column_name}: {e}")  # 디버깅용 출력
        return [-1.0] * len(column_data)  # 기본값 할당


# def preprocess_augmented_data(augmented_data, encoders):
#     try:
#         numeric_data = augmented_data.copy()
#
#         for column, encoder in encoders.items():
#             if column in numeric_data.columns:
#                 try:
#                     col_data = numeric_data[[column]]  # DataFrame 형태로 유지
#                     numeric_data[column] = encoder.transform(col_data)
#                 except Exception as e:
#                     logging.error(f"Error encoding column {column}: {e}")
#                     numeric_data[column] = -1  # 실패 시 -1 할당
#
#         logging.info("Preprocessed augmented_data:")
#         logging.info(numeric_data.dtypes)
#         logging.info(numeric_data.head())
#
#         return numeric_data
#     except Exception as e:
#         raise ValueError(f"Error in preprocess_augmented_data: {e}")


COLUMN_MAPPING = {
    '장애유형': 'disType',
    '학수번호': 'haksuNum',
    '분반': 'classNum',
    '교과목명': 'className',
    '개설학과/전공': 'classDept',
    '교과영역': 'subjectArea',
    '교과목 구분': 'subjectArea2',
    '교수명': 'profName',
    '정원': 'studentNum',
    '시간': 'classTime',
    '학점': 'Credit',
    '요일/교시': 'classDate',
    '교실': 'classLocation',
    '지원종류': 'helpType',
    '원격강의': 'classOnline',
    '이동지원': 'helpMove',
    '교생실습지원': 'helpTeach',
    '수업참여지원': 'helpParticipate',
    '알림사항지원': 'helpAlert',
    '결과': 'result'
}


# 증강 데이터 전처리
# def preprocess_augmented_data(augmented_data, encoders):
#     """
#     증강 데이터를 수치형으로 변환
#     """
#     try:
#         # COLUMN_MAPPING을 사용해 열 이름 변경
#         augmented_data.rename(columns=COLUMN_MAPPING, inplace=True)
#
#         numeric_data = augmented_data.copy()
#
#         # 각 컬럼을 인코더를 사용해 변환
#         for column, encoder in encoders.items():
#             if column not in numeric_data.columns:
#                 logging.warning(f"Column '{column}' not found in augmented_data.")
#                 continue
#
#             logging.info(f"Encoding start: {column}")
#             try:
#                 # 변환 전 데이터
#                 logging.info(f"Data before encoding in '{column}':\n{numeric_data[[column]].head()}")
#
#                 # OrdinalEncoder를 사용하여 변환
#                 transformed = encoder.transform(numeric_data[[column]])
#                 numeric_data[column] = transformed
#
#                 # 변환 후 데이터
#                 logging.info(f"Transformed data for '{column}':\n{numeric_data[[column]].head()}")
#             except Exception as e:
#                 logging.error(f"Error encoding column '{column}': {e}")
#                 numeric_data[column] = -1  # 변환 실패 시 기본값 -1 할당
#
#         # 디버깅: 최종 데이터 확인
#         logging.info("Preprocessed augmented_data:")
#         logging.info(numeric_data.head())
#         return numeric_data
#
#     except Exception as e:
#         raise ValueError(f"Error in preprocess_augmented_data: {e}")
def preprocess_augmented_data(augmented_data, encoders):
    """
    증강 데이터를 수치형으로 변환
    """
    try:
        # COLUMN_MAPPING을 사용해 열 이름 변경
        augmented_data.rename(columns=COLUMN_MAPPING, inplace=True)

        numeric_data = augmented_data.copy()

        # 문자형 데이터를 숫자로 변환 ('O' → 1, 'X' → 0)
        replacement_map = {'O': 1, 'X': 0}
        for column in ['classOnline', 'helpMove', 'helpTeach', 'helpParticipate', 'helpAlert', 'result']:
            if column in numeric_data.columns:
                numeric_data[column] = numeric_data[column].replace(replacement_map)

        # 각 컬럼을 인코더를 사용해 변환
        for column, encoder in encoders.items():
            if column not in numeric_data.columns:
                logging.warning(f"Column '{column}' not found in augmented_data.")
                continue

            logging.info(f"Encoding start: {column}")
            try:
                # 변환 전 데이터
                logging.info(f"Data before encoding in '{column}':\n{numeric_data[[column]].head()}")

                # OrdinalEncoder를 사용하여 변환
                transformed = encoder.transform(numeric_data[[column]])
                numeric_data[column] = transformed

                # 변환 후 데이터
                logging.info(f"Transformed data for '{column}':\n{numeric_data[[column]].head()}")
            except Exception as e:
                logging.error(f"Error encoding column '{column}': {e}")
                numeric_data[column] = -1  # 변환 실패 시 기본값 -1 할당

        # 디버깅: 최종 데이터 확인
        logging.info("Preprocessed augmented_data:")
        logging.info(numeric_data.head())
        return numeric_data

    except Exception as e:
        raise ValueError(f"Error in preprocess_augmented_data: {e}")


# 요청 데이터 전처리
def preprocess_input(data):
    try:
        logging.info(f"Raw input data: {data}")
        dis_type = data.get("disType", "")
        class_dept = data.get("major", "")
        help_type = data.get("help_type", "")

        dis_type_encoded = transform_column("disType", [dis_type], ordinal_encoders['disType'])[0]
        class_dept_encoded = transform_column("classDept", [class_dept], ordinal_encoders['classDept'])[0]
        help_type_encoded = transform_column("helpType", [help_type], ordinal_encoders['helpType'])[0]

        input_data = [
            dis_type_encoded, 0, 0, 0, class_dept_encoded, 0, 0, 0, 0, 0, 0, 0, 0, help_type_encoded, 0, 0, 0, 0, 0
        ]
        feature_names = list(model.feature_names_in_)
        input_df = pd.DataFrame([input_data], columns=feature_names)
        logging.info(f"Processed input data: {input_df}")
        return input_df
    except Exception as e:
        raise ValueError(f"Data preprocessing failed: {e}")


# 매칭 확률 예측 함수
def predict_match_probability(data):
    try:
        prediction = model.predict_proba(data)
        match_probability = prediction[0][1]
        return match_probability
    except Exception as e:
        raise ValueError(f"Model prediction failed: {e}")


def transform_column(column_name, column_data, encoder):
    """
    OrdinalEncoder로 변환 실패 시 기본값(-1) 반환
    """
    try:
        column_df = pd.DataFrame(column_data, columns=[column_name])
        encoded_data = encoder.transform(column_df).flatten()
        logging.info(f"Successfully encoded column '{column_name}': {encoded_data[:5]}")
        return encoded_data
    except ValueError as e:
        logging.error(f"ValueError in {column_name}: {e}")
        return [-1.0] * len(column_data)  # 기본값 할당

# 매칭 계산 함수
def find_best_match(request_data, augmented_data, encoders):
    try:
        # 증강 데이터를 수치형으로 변환
        numeric_augmented_data = preprocess_augmented_data(augmented_data, encoders)

        # 'result' 열을 제외하고 비교에 사용할 열만 선택
        comparison_columns = [col for col in numeric_augmented_data.columns if col != "result"]
        numeric_augmented_data = numeric_augmented_data[comparison_columns]

        # 요청 데이터를 numpy 배열로 변환
        request_vector = request_data.to_numpy()
        augmented_vectors = numeric_augmented_data.to_numpy()

        # 코사인 유사도 계산
        similarities = cosine_similarity(request_vector, augmented_vectors)

        # 최고 유사도를 가진 데이터 인덱스 및 유사도 추출
        best_match_idx = np.argmax(similarities[0])
        highest_similarity = similarities[0][best_match_idx]
        best_match = augmented_data.iloc[best_match_idx]  # 원본 데이터에서 매칭된 행 반환

        return best_match, highest_similarity
    except Exception as e:
        raise ValueError(f"Error in find_best_match: {e}")


# Flask 라우트 설정
@match_bp.route('/course', methods=['POST'])
def course_match_result():
    if not request.is_json:
        return jsonify({"error": "Invalid content type. Expected JSON."}), 400

    try:
        match_request = request.get_json()
        processed_request = preprocess_input(match_request)

        augmented_data = pd.read_csv(ctgan_data_path)
        best_match, match_probability = find_best_match(processed_request, augmented_data, ordinal_encoders)

        return jsonify({
            "match_probability": match_probability,
            "best_match": best_match.to_dict()
        }), 200
    except Exception as e:
        logging.error(f"Error in course_match_result: {e}")
        return jsonify({"error": str(e)}), 500

# 초기화
initialize_encoders()