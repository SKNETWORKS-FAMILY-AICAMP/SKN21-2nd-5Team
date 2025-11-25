import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from admin_util import visualize_customer_data, get_customer_prediction, display_prediction_results

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'test.csv')
PREDICTIONS_PATH = os.path.join(BASE_DIR, 'data', 'test_predictions.csv')

customer_df = pd.read_csv(DATA_PATH)
predictions_df = pd.read_csv(PREDICTIONS_PATH)

# 파일 존재 여부 확인
if not os.path.exists(DATA_PATH):
    st.error(f"❌ 파일을 찾을 수 없습니다: {DATA_PATH}")
    st.stop()

with st.sidebar:
    st.title("관리자 페이지")
    st.markdown("-----")
    st.page_link(page="http://localhost:8504/", label="메인 통계 대시보드", icon="📊")
    st.page_link(page="http://localhost:8504/", label="개별 고객 분석", icon="👤")  # 현재 페이지는 비활성화
   
st.title("🏨 개별 고객 정보 분석")
st.markdown("---")

# 고객 선택 방법
col1, col2 = st.columns(2)
with col1:
    st.markdown("##### ✅ 고객 선택 방식")
with col2:
    selection_method = st.radio(
        "선택 방식", 
        ["테이블에서 행 선택", "고객 ID로 직접 검색"], 
        label_visibility="collapsed", 
        horizontal=True
    )
st.markdown("<br>", unsafe_allow_html=True)

# 검색 방식 1 - 고객 ID로 직접 검색
if selection_method == "고객 ID로 직접 검색":
    customer_ids = customer_df['name'].unique() if 'name' in customer_df.columns else []
    selected_customer_id = st.selectbox("분석할 고객 이름 선택",options=customer_ids, index=None)
    
    if selected_customer_id:
        selected_row = customer_df[customer_df['name'] == selected_customer_id].iloc[0]
        selected_index = customer_df[customer_df['name'] == selected_customer_id].index[0]
        
        # 고객 기본 정보 표시
        visualize_customer_data(selected_row, selected_index)
        
        # 예측 결과 분석
        prediction_data = get_customer_prediction(selected_row, predictions_df)
        display_prediction_results(prediction_data, predictions_df)

# 검색 방식 2 - 데이터 테이블에서 행 선택
else:
    st.subheader("🗂️ 고객 데이터 테이블")
    st.write("아래에서 행을 선택하여 상세 정보를 확인하세요")
    
    # 인터랙티브 데이터프레임 (행 선택 가능)
    events = st.dataframe(
        data=customer_df,
        width='stretch',
        on_select="rerun",
        selection_mode="single-row"
    )
    
    # 행이 선택되었을 때 시각화
    if events.selection["rows"]:
        selected_indices = events.selection["rows"]
        selected_index = selected_indices[0]
        selected_row = customer_df.iloc[selected_index]
        
        # 고객 기본 정보 표시
        visualize_customer_data(selected_row, selected_index)
        
        # 예측 결과 분석
        prediction_data = get_customer_prediction(selected_row, predictions_df)
        display_prediction_results(prediction_data, predictions_df)

