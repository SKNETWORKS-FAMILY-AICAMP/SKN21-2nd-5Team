import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
pd.set_option('display.max_columns', None)  #

def load_data():
    """
    현재 스크립트의 위치를 기준으로 ../data/hotel_bookings.csv 파일을 로드합니다.
    """
    # 현재 파일(train.py)의 절대 경로 디렉토리
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 상위 폴더(..)로 이동 후 data 폴더의 파일 지정
    file_path = os.path.join(current_dir, '..', 'data', 'hotel_bookings.csv')
    file_path = os.path.normpath(file_path)
    
    # 파일을 찾을 수 없는 경우를 대비한 예외 처리
    if not os.path.exists(file_path):
        # 현재 디렉토리가 아닌 다른 경로에 파일이 있을 수 있으므로 상대 경로를 바로 시도 (노트북 환경 등)
        try:
            df = pd.read_csv('../data/hotel_bookings.csv')
            print(f"Reading data from relative path: ../data/hotel_bookings.csv")
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Error: 파일을 지정된 경로에서 찾을 수 없습니다. (경로 확인 필요)")
            return None

    try:
        df = pd.read_csv(file_path)
        print(f"Reading data from: {file_path}")
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def preprocess_data(df):
    """
    XGBoost 모델 학습을 위한 개선된 전처리 함수
    - 데이터 유출 컬럼 및 중복/ID 컬럼 제거
    - 이상치 처리 (Capping)
    - 결측치 처리 (Imputation)
    - One-Hot Encoding
    """
    df_proc = df.copy()
    # 필요없는 컬럼 제거
    useless_col = ['days_in_waiting_list', 'arrival_date_year', 'assigned_room_type', 'booking_changes',
               'reservation_status', 'country', 'reservation_status_date']

    df_proc.drop(useless_col, axis = 1, inplace = True)
    
    # ADR (Average Daily Rate): 99th percentile로 capping
    adr_99 = df_proc['adr'].quantile(0.99)
    df_proc['adr'] = np.where(df_proc['adr'] > adr_99, adr_99, df_proc['adr'])
    
    # === Feature Engineering ===
    # 1. has_special_requests: 특별 요청 있음 여부
    df_proc['has_special_requests'] = (df_proc['total_of_special_requests'] > 0).astype(int)
    
    # 2. is_planned: 사전 계획 예약 (90일 이상)
    df_proc['is_planned'] = (df_proc['lead_time'] >= 90).astype(int)
    
    # 3. is_last_minute: 당일/직전 예약 (3일 이내)
    df_proc['is_last_minute'] = (df_proc['lead_time'] <= 3).astype(int)
    
    # 4. needs_parking: 주차 필요 여부
    df_proc['needs_parking'] = (df_proc['required_car_parking_spaces'] > 0).astype(int)
    
    print(f"\n✅ Feature Engineering 완료: 4개의 새로운 피처 추가")
    print(f"   - has_special_requests, is_planned, is_last_minute, needs_parking")

    
    num_cols = df_proc.select_dtypes(include=[np.number]).columns
    cat_cols = df_proc.select_dtypes(include=['object']).columns
    
    # 숫자형 결측치 처리: 0으로 대체 (children, adr 등)
    df_proc[num_cols] = df_proc[num_cols].fillna(0)
    
    # 범주형 결측치 처리: 'Unknown'으로 대체 (country, meal 등)
    df_proc[cat_cols] = df_proc[cat_cols].fillna('Unknown')
    
    # 원-핫 인코딩 전 컬럼 출력
    print(f"\n원-핫 인코딩 전 컬럼 ({len(df_proc.columns)}개):")
    print(df_proc.columns.tolist())
    print(f"\n원-핫 인코딩 대상 범주형 컬럼 ({len(cat_cols)}개):")
    print(cat_cols.tolist())
    
    # 4. 범주형 변수 인코딩 (One-Hot Encoding)
    cat_cols_for_ohe = df_proc.select_dtypes(include=['object', 'category']).columns
    df_proc = pd.get_dummies(df_proc, columns=cat_cols_for_ohe, drop_first=True)
    
    print(f"\n원-핫 인코딩 후 컬럼 ({len(df_proc.columns)}개):")
    print(df_proc.columns.tolist())
    
    return df_proc

def train_model(df):
    """
    데이터를 분할하고 XGBoost 모델을 학습/평가합니다.
    - Train Set의 비율만을 사용하여 scale_pos_weight를 계산합니다 (데이터 유출 방지).
    """
    # 1. X, y 분리
    target = 'is_canceled'
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")
        
    X = df.drop(target, axis=1)
    y = df[target]
    
    # 2. Train / Test Split (80:20, 계층 샘플링)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # 학습에 사용되는 컬럼 출력
    print(f"\n학습에 사용되는 컬럼 ({len(X_train.columns)}개):")
    print(X_train.columns.tolist())

    from catboost import CatBoostClassifier

    cat = CatBoostClassifier(iterations=100, verbose=0)
    cat.fit(X_train, y_train)
    y_pred_cat = cat.predict(X_test)
    acc_cat = accuracy_score(y_test,y_pred_cat)
    conf = confusion_matrix(y_test, y_pred_cat)
    clf_report = classification_report(y_test, y_pred_cat)

    print("accuracy : ", acc_cat)
    print("conf :", conf)
    print("report :", clf_report)
    
    # 모델 저장
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, '..', 'model')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'catboost_model.cbm')
    cat.save_model(model_path)
    print(f"\n모델 저장 완료: {model_path}")
    
    # 학습에 사용한 feature 컬럼 저장
    import pickle
    feature_cols_path = os.path.join(model_dir, 'feature_columns.pkl')
    with open(feature_cols_path, 'wb') as f:
        pickle.dump(X_train.columns.tolist(), f)
    print(f"Feature 컬럼 저장 완료: {feature_cols_path}")
    
    return cat

if __name__ == "__main__":
    # 1. 데이터 로드
    raw_data = load_data()
    
    if raw_data is not None:
        # 2. 전처리
        print("\nStarting preprocessing...")
        processed_data = preprocess_data(raw_data)
        print(f"Preprocessing completed. Final shape: {processed_data.shape}")
        
        # 3. 학습 및 평가
        model = train_model(processed_data)
        print("\n학습 및 모델 저장이 완료되었습니다.")