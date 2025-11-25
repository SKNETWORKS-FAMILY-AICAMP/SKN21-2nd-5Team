import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys # sys 모듈 임포트

# --- 경로 설정 및 sys.path 추가 ---
# 현재 스크립트가 있는 디렉토리 (streamlit/pages/)
current_pages_dir = os.path.dirname(os.path.abspath(__file__))

# 'streamlit' 디렉토리의 절대 경로 (pages의 한 단계 상위)
streamlit_root_dir = os.path.dirname(current_pages_dir)

# 'streamlit' 디렉토리를 Python의 모듈 검색 경로(sys.path)에 추가합니다.
# 이렇게 하면 'streamlit' 디렉토리 바로 아래에 있는 모듈(예: admin_util.py, utils.py)을 직접 임포트할 수 있습니다.
if streamlit_root_dir not in sys.path:
    sys.path.insert(0, streamlit_root_dir)

# --- 유틸리티 함수 임포트 ---
# 'streamlit_root_dir'가 sys.path에 추가되었으므로 'admin_util'과 'utils'를 직접 임포트합니다.
from admin_util import visualize_customer_data, get_customer_prediction, display_prediction_results

# 이미지 상 'utils.py' 파일이 'streamlit' 폴더 바로 아래에 있을 것으로 예상됩니다.
# 만약 utils가 폴더(패키지)이고 그 안에 check_access 함수가 있다면 'from utils import check_access' 또는 'from utils.access_control import check_access' 등이 될 수 있습니다.
from utils import check_access, display_access_denied_message_once


current_page_name = os.path.basename(__file__) # 현재 페이지 스크립트 이름 (예: "admin_pg.py")

display_access_denied_message_once(current_page_name)

# 페이지 로드 시 가장 먼저 접근 권한 확인
# check_access 함수 호출 시 두 번째 인자로 current_page_name을 전달합니다.
check_access("admin", current_page_name) # 이 페이지는 "admin"만 접근 가능

# --- 데이터 파일 경로 설정 ---
# 이미지 상 'data' 폴더가 'streamlit' 폴더 바로 아래에 있는 것으로 보입니다.
DATA_PATH = os.path.join(streamlit_root_dir, '..', 'data', 'test.csv')
PREDICTIONS_PATH = os.path.join(streamlit_root_dir, '..', 'data', 'test_predictions.csv')

# 파일 존재 여부 확인 및 로드
if not os.path.exists(DATA_PATH):
    st.error(f"❌ 고객 데이터를 찾을 수 없습니다: {DATA_PATH}")
    st.stop()
if not os.path.exists(PREDICTIONS_PATH):
    st.error(f"❌ 예측 데이터를 찾을 수 없습니다: {PREDICTIONS_PATH}")
    st.stop()

customer_df = pd.read_csv(DATA_PATH)
predictions_df = pd.read_csv(PREDICTIONS_PATH)

with st.sidebar:
    st.title("관리자 페이지")
    st.markdown("-----")
    # Streamlit Multipage 앱에서는 st.page_link를 사용하여 다른 페이지로 이동합니다.
    # http://localhost:8504/ 는 앱의 루트를 나타냅니다. 실제 경로에 따라 수정 필요할 수 있습니다.
    st.page_link("main.py", label="메인 통계 대시보드", icon="📊") # main.py가 메인 페이지라고 가정
    st.page_link("pages/admin_pg.py", label="개별 고객 분석", icon="👤")  # 현재 페이지는 활성화

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

    selected_customer_id = st.selectbox("분석할 고객 이름 선택", options=customer_ids, index=None)

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
