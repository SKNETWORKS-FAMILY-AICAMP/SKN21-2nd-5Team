# 호텔 예약 취소 예측 모델

## SKN21-2nd-5Team

<div align="center">


</div>

## 팀원
| 전우영 | 이성진 | 최자슈아주원 | 유성현 | 진승언 |
|:------:|:------:|:------:|:------:|:------:|

----

## 프로젝트 소개
본 프로젝트는 호텔 예약 데이터를 기반으로 머신러닝을 활용하여 고객의 예약 취소 가능성을 예측하고, 예측 결과를 실시간으로 확인할 수 있는 대시보드 시스템을 구현한 프로젝트입니다.

## 프로젝트 목표
- 호텔 예약 취소 예측 모델 개발
- LightGBM 기반 고성능 분류 모델 구현
- Streamlit을 활용한 실시간 예측 시스템 구축
- 고객 맞춤형 예약 관리 대시보드 제공

----

## 데이터셋

### 데이터 출처
이 데이터셋은 시티 호텔과 리조트 호텔의 예약 정보를 포함하고 있으며, 예약 시기, 숙박 기간, 성인/어린이/유아 수, 주차 공간 수 등의 정보가 담겨 있습니다.

### 주요 특징
- **총 데이터 수**: 약 119,390건
- **호텔 유형**: City Hotel, Resort Hotel
- **예측 목표**: 예약 취소 여부 (is_canceled)

----

## 기술 스택

**Libraries**

![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![numpy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

**Environment & Tools**

![VSCode](https://img.shields.io/badge/VSCode-0078D4?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

----
## 프로젝트 구조

```
SKN21-2nd-5Team/
├── data/                    # 데이터 파일
│   ├── hotel_bookings.csv
│   └── images/
├── modeling/                # 모델 학습 코드
│   ├── v1.py               # LightGBM 학습
│   └── predict.py          # 예측 함수
├── streamlit/              # Streamlit 앱
│   ├── pages/
│   │   └── guest_pg.py     # 사용자 페이지
│   └── main.py
├── model/                  # 저장된 모델
│   ├── lgbm_model.txt
│   └── feature_columns.pkl
├── EDA/                    # 탐색적 데이터 분석
└── README.md
```

----
## 데이터 전처리

### 1. 피처 엔지니어링
- **가족 여부 판별**: `is_family` - 어린이나 유아가 있는 경우
- **예약 리드타임 그룹화**: `lead_time_group` - 당일/단기/중기/장기 예약 구분
- **국가 그룹화**: `country_grouped` - 주요 국가 vs 기타 국가
- **특별 요청 여부**: `has_special_requests` - 특별 요청 존재 여부
- **주차 필요 여부**: `needs_parking` - 주차 공간 필요 여부

### 2. 데이터 정제
- 데이터 누수 컬럼 제거: `reservation_status_date`
- 중복 정보 컬럼 제거: `arrival_date_month`
- 결측치 처리: 수치형(0), 범주형('Unknown')
- One-Hot Encoding 적용: 범주형 변수 변환

### 3. 데이터 분할
- Train/Validation: 80/20 비율
- Stratified Split 적용 (클래스 불균형 고려)

----

## 모델 구현

### LightGBM 모델
대부분 정형 데이터이기에 Tree based 모델인 XGBoost, LGBM, CatBoost 등 중에서 baseline 성능이 가장 좋은 모델을 선택 
  
### Optuna를 통한 자동하이퍼 파라미터 튜닝  
n_estimators, learning_rate, max_depth 등의 하이퍼파라미터들을 optuna 라이브러리를 통해 자동으로 찾아 최적화

### 모델 최종 성능

```
----------------------------------------
Model Accuracy: 0.8929
----------------------------------------
Classification Report:

              precision    recall  f1-score   support

           0       0.92      0.91      0.91     22550
           1       0.85      0.87      0.86     13267

    accuracy                           0.89     35817
   macro avg       0.88      0.89      0.89     35817
weighted avg       0.89      0.89      0.89     35817

=== 추가 성능 지표 ===
AUC-ROC: 0.9598
Recall: 0.8658
F1-Score: 0.8570
```

----

## Streamlit 활용 시스템 구현

### 1. 사용자 호텔 예약 페이지
- 개인 예약 정보 입력

### 2. 관리자 대시보드
- 예약 현황 통계
- 취소율 분석
- 시각화 차트 제공
- 개인 예약 내역 조회 가능
----

## 기대효과

### 1. 고객 관리
- 취소 위험 고객 대상 프로모션
- 고객 특별 요청 사항 모니터링 및 개선  

### 2. 운영 효율화
- 데이터 기반 의사결정
- 자원 배분 최적화
- 실시간 예측 시스템 활용

----

## 한계점 및 개선 가능성

### 한계점
- 외부 요인 (경기 상황, 계절성) 미반영
- 실시간 데이터 업데이트 필요
- 모델 버전 호환성 이슈 (LightGBM)

### 개선 가능성
1. **모델 개선**
   - 앙상블 모델 적용
   - 추가 피처 엔지니어링

2. **데이터 확장**
   - 특별 요청 사항과 같은 자연어 데이터 사용
----


## 실행 방법

### 1. 환경 설정
```bash
pip install pandas numpy scikit-learn lightgbm streamlit
```

### 2. 모델 학습
```bash
python modeling/v1.py
```

### 3. Streamlit 앱 실행
```bash
streamlit run streamlit/main.py
```

----

## 한 줄 회고

> #### 전우영
> 
---

> #### 이성진
> 
---

> #### 최자슈아주원
> 
---

> #### 유성현
> 

---

> #### 진승언
> 
---
  