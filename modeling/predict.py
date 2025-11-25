import os
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import warnings

# joblib 경고 무시
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # CPU 코어 수 설정

def load_test_data():
    """
    test.csv 파일을 로드합니다.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'data', 'test.csv')
    file_path = os.path.normpath(file_path)
    
    try:
        df = pd.read_csv(file_path)
        print(f"Test data loaded from: {file_path}")
        print(f"Test data shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: 파일을 찾을 수 없습니다. 경로: {file_path}")
        return None

def preprocess_test_data(df):
    """
    test.csv를 v1.py와 동일한 방식으로 전처리합니다.
    """
    df_proc = df.copy()
    
    # client_id 컬럼을 별도로 저장 (예측 결과에 포함시킬 용도)
    client_ids = None
    if 'client_id' in df_proc.columns:
        client_ids = df_proc['client_id'].copy()
        df_proc.drop(['client_id'], axis=1, inplace=True)
    
    # reservation_status_date는 데이터 누수 컬럼이므로 제거
    df_proc.drop(['reservation_status_date'], axis=1, inplace=True, errors='ignore')
    df_proc.drop(['arrival_date_month'], axis=1, inplace=True, errors='ignore')
    
    # 결측치 처리
    num_cols = df_proc.select_dtypes(include=[np.number]).columns
    cat_cols = df_proc.select_dtypes(include=['object']).columns
    
    df_proc[num_cols] = df_proc[num_cols].fillna(0)
    df_proc[cat_cols] = df_proc[cat_cols].fillna('Unknown')
    
    # One-Hot Encoding
    cat_cols_for_ohe = df_proc.select_dtypes(include=['object', 'category']).columns
    df_proc = pd.get_dummies(df_proc, columns=cat_cols_for_ohe, drop_first=True)
    
    return df_proc, client_ids

def load_model():
    """
    저장된 LightGBM 모델을 로드합니다.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'model', 'lgbm_model.txt')
    model_path = os.path.normpath(model_path)
    
    if not os.path.exists(model_path):
        print(f"Error: 모델 파일을 찾을 수 없습니다. 경로: {model_path}")
        print("먼저 v1.py를 실행하여 모델을 학습시켜주세요.")
        return None
    
    try:
        model = lgb.Booster(model_file=model_path)
        print(f"모델 로드 완료: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def align_test_features(X_test, feature_cols):
    """
    테스트 데이터의 컬럼을 학습 시 사용한 컬럼과 동일하게 맞춥니다.
    """
    # 학습 시 없던 컬럼 제거
    for col in X_test.columns:
        if col not in feature_cols:
            X_test = X_test.drop(col, axis=1)
    
    # 학습 시 있던 컬럼 중 테스트에 없는 컬럼은 0으로 채움
    for col in feature_cols:
        if col not in X_test.columns:
            X_test[col] = 0
    
    # 컬럼 순서를 학습 시와 동일하게 정렬
    X_test = X_test[feature_cols]
    
    return X_test

def predict_test_data(model, X_test):
    """
    테스트 데이터에 대해 예측을 수행합니다.
    """
    try:
        probabilities = model.predict(X_test)  # 확률값 반환
        predictions = (probabilities > 0.5).astype(int)  # 0.5 기준 이진 분류
        print(f"\n예측 완료!")
        print(f"예측 결과 샘플 (첫 10개):")
        print(f"{'Index':<10}{'Prediction':<15}{'Prob(Cancel)':<20}")
        print("-" * 45)
        for i in range(min(10, len(predictions))):
            print(f"{i:<10}{predictions[i]:<15}{probabilities[i]:<20.4f}")
        return predictions, probabilities
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

if __name__ == "__main__":
    print("=" * 60)
    print("테스트 데이터 예측 시작")
    print("=" * 60)
    
    # 1. 테스트 데이터 로드
    test_data = load_test_data()
    if test_data is None:
        exit(1)
    
    # 2. 전처리
    print("\n전처리 시작...")
    # is_canceled 컬럼이 있다면 제거 (예측 대상이므로)
    if 'is_canceled' in test_data.columns:
        test_data = test_data.drop('is_canceled', axis=1)
    
    processed_test, client_ids = preprocess_test_data(test_data)
    print(f"전처리 완료. Shape: {processed_test.shape}")
    print(f"컬럼: {processed_test.columns.tolist()}")
    
    # 3. 모델 및 feature 컬럼 로드
    print("\n모델 로드 중...")
    model = load_model()
    if model is None:
        exit(1)
    
    # Feature 컬럼 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    feature_cols_path = os.path.join(current_dir, '..', 'model', 'feature_columns.pkl')
    feature_cols_path = os.path.normpath(feature_cols_path)
    
    try:
        with open(feature_cols_path, 'rb') as f:
            feature_cols = pickle.load(f)
        print(f"Feature 컬럼 로드 완료: {len(feature_cols)}개")
    except FileNotFoundError:
        print(f"Error: Feature 컬럼 파일을 찾을 수 없습니다. 경로: {feature_cols_path}")
        print("먼저 v1.py를 실행하여 모델을 학습시켜주세요.")
        exit(1)
    
    # 테스트 데이터 컬럼 정렬
    print("\n테스트 데이터 컬럼을 학습 시와 동일하게 정렬 중...")
    processed_test = align_test_features(processed_test, feature_cols)
    print(f"정렬 완료. Shape: {processed_test.shape}")
    
    # 4. 예측
    print("\n예측 수행 중...")
    predictions, probabilities = predict_test_data(model, processed_test)
    
    if predictions is not None:
        # 결과를 CSV로 저장
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(current_dir, '..', 'data', 'test_predictions.csv')
        output_path = os.path.normpath(output_path)
        
        # 예측 결과를 name, prediction, probability_no_cancel, probability_cancel 형식으로 저장
        # client_id가 이름 역할을 한다고 가정
        # name 컬럼이 있으면 예측 결과에 사용
        if 'name' in test_data.columns:
            name_col = test_data['name'].copy()
            test_data = test_data.drop(['name'], axis=1)
        elif 'client_id' in test_data.columns:
            name_col = test_data['client_id'].copy()
            test_data = test_data.drop(['client_id'], axis=1)
        else:
            name_col = pd.Series(range(len(test_data)))
        result_df = pd.DataFrame({
            'name': name_col.values,
            'prediction': predictions,
            'probability_no_cancel': 1 - probabilities,
            'probability_cancel': probabilities
        })
    result_df.to_csv(output_path, index=False)
    print(f"\n예측 결과가 저장되었습니다: {output_path}")

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)
