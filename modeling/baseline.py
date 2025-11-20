import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    """
    현재 스크립트(train.py)의 위치를 기준으로 ../data/hotel_bookings.csv 파일을 로드합니다.
    """
    # 현재 파일(train.py)의 절대 경로 디렉토리 (modeling 폴더)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 상위 폴더(..)로 이동 후 data 폴더의 파일 지정
    file_path = os.path.join(current_dir, '..', 'data', 'hotel_bookings.csv')
    file_path = os.path.normpath(file_path)
    print(f"Reading data from: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("Error: 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return None

def preprocess_data(df):
    """
    Baseline 전처리 함수
    - Data Leakage 컬럼 제거
    - 결측치 처리
    - One-Hot Encoding
    """
    df_proc = df.copy()
    
    # 1. Data Leakage 방지: 예약 상태 관련 컬럼 제거
    leakage_cols = ['reservation_status', 'reservation_status_date']
    df_proc = df_proc.drop(columns=leakage_cols, errors='ignore')
    
    # 2. 결측치 처리 (Baseline: 숫자=0, 문자='Unknown')
    num_cols = df_proc.select_dtypes(include=[np.number]).columns
    cat_cols = df_proc.select_dtypes(include=['object']).columns
    
    # SettingWithCopyWarning 방지를 위해 딕셔너리 형태로 fillna 하거나 직접 할당
    df_proc[num_cols] = df_proc[num_cols].fillna(0)
    df_proc[cat_cols] = df_proc[cat_cols].fillna('Unknown')
    
    # 3. 범주형 변수 인코딩 (One-Hot Encoding)
    # drop_first=True로 다중공선성 문제 완화
    df_proc = pd.get_dummies(df_proc, columns=cat_cols, drop_first=True)
    
    return df_proc

def train_model(df):
    # 1. X, y 분리
    target = 'is_canceled'
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")
        
    X = df.drop(target, axis=1)
    y = df[target]
    
    # 2. Train / Test Split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # 3. XGBoost 모델 정의 (Baseline 하이퍼파라미터)
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1  # 가능한 모든 CPU 코어 사용
    )
    
    # 4. 학습
    print("\nStarting training...")
    model.fit(X_train, y_train)
    print("Training completed.")
    
    # 5. 예측 및 평가
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("-" * 40)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("-" * 40)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))
    
    return model

if __name__ == "__main__":
    # 데이터 로드
    raw_data = load_data()
    
    if raw_data is not None:
        # 전처리
        processed_data = preprocess_data(raw_data)
        
        # 학습 및 평가
        model = train_model(processed_data)