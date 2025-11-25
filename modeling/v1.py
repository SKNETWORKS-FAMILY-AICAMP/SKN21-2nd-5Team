import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import warnings

# joblib 경고 무시
warnings.filterwarnings('ignore', category=UserWarning)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
pd.set_option('display.max_columns', None)  #

def load_data():
    """
    현재 스크립트의 위치를 기준으로 ../data/hotel_bookings.csv 파일을 로드합니다.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'data', 'hotel_bookings.csv')
    file_path = os.path.normpath(file_path)
    
    # 파일을 찾을 수 없는 경우를 대비한 예외 처리
    if not os.path.exists(file_path):
        # 현재 디렉토리가 아닌 다른 경로에 파일이 있을 수 있으므로 상대 경로를 바로 시도 (노트북 환경 등)
        try:
            df = pd.read_csv('../data/hotel_bookings.csv')
            return df
        except FileNotFoundError:
            print(f"Error: 파일을 지정된 경로에서 찾을 수 없습니다. (경로 확인 필요)")
            return None

    try:
        df = pd.read_csv(file_path)
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
    
    # 1. country 결측치 처리
    mfc = df_proc['country'].mode()[0]
    df_proc['country'] = df_proc['country'].fillna(mfc)
    
    # 2. children 결측치 처리
    df_proc['children'] = df_proc['children'].fillna(0).astype(int)
    
    # 3. ADR 이상치 제거 (5000 초과, 음수)
    df_proc = df_proc[df_proc['adr'] <= 5000].copy()
    df_proc['adr'] = df_proc['adr'].apply(lambda x: x if x >= 0 else 0)
    
    print(f"\n데이터 클리닝 완료. Shape: {df_proc.shape}")
    
    useless_col = ['reservation_status', 'reservation_status_date']

    df_proc.drop(useless_col, axis = 1, inplace = True)
    
    num_cols = df_proc.select_dtypes(include=[np.number]).columns
    cat_cols = df_proc.select_dtypes(include=['object']).columns
    
    # 숫자형 결측치 처리: 0으로 대체 (children, adr 등)
    df_proc[num_cols] = df_proc[num_cols].fillna(0)
    
    # 범주형 결측치 처리: 'Unknown'으로 대체 (country, meal 등)
    df_proc[cat_cols] = df_proc[cat_cols].fillna('Unknown')
    
    # 4. 범주형 변수 인코딩 (One-Hot Encoding)
    cat_cols_for_ohe = df_proc.select_dtypes(include=['object', 'category']).columns
    df_proc = pd.get_dummies(df_proc, columns=cat_cols_for_ohe, drop_first=True)
    

    return df_proc

def train_model(df):
    """
    데이터를 분할하고 Optuna로 LightGBM 파라미터 튜닝 및 학습/평가합니다.
    """
    target = 'is_canceled'
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    scale_pos_weight = neg / pos

    def objective(trial):
        param = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': 42,
            'scale_pos_weight': scale_pos_weight,
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0)
        }
        model = lgb.LGBMClassifier(**param)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, y_pred_proba)
        return auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print('Best trial:', study.best_trial.params)

    best_params = study.best_trial.params
    best_params.update({
        'random_state': 42,
        'scale_pos_weight': scale_pos_weight,
        'verbosity': -1
    })
    lgbm = lgb.LGBMClassifier(**best_params)
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_test)
    y_pred_proba = lgbm.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    conf = confusion_matrix(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)

    print("accuracy : ", acc)
    print("conf :", conf)
    print("report :", clf_report)

    from sklearn.metrics import roc_auc_score, recall_score, f1_score
    print(f"\n=== 추가 성능 지표 ===")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, '..', 'model')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'lgbm_model.txt')
    lgbm.booster_.save_model(model_path)

    import pickle
    feature_cols_path = os.path.join(model_dir, 'feature_columns.pkl')
    with open(feature_cols_path, 'wb') as f:
        pickle.dump(X_train.columns.tolist(), f)
    print(f"Feature 컬럼 저장 완료: {feature_cols_path}")

    return lgbm

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