import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from utils import check_access, display_access_denied_message_once




current_page_name = os.path.basename(__file__) # 현재 페이지 스크립트 이름 (예: "index.py")

display_access_denied_message_once(current_page_name)

# 페이지 로드 시 가장 먼저 접근 권한 확인
# check_access 함수 호출 시 두 번째 인자로 current_page_name을 전달합니다.
check_access("admin", current_page_name) # 이 페이지는 "admin"만 접근 가능


st.set_page_config(
    page_title="호텔 예약 취소 위험도 관리",
    page_icon="🏨",
    layout="wide"
)

# 데이터 로드 함수
@st.cache_data
def load_predictions():
    """test_predictions.csv 파일을 로드합니다."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', '..', 'data', 'test_predictions.csv')
    file_path = os.path.normpath(file_path)
    
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"예측 결과 파일을 찾을 수 없습니다: {file_path}")
        st.info("먼저 modeling/test.py를 실행하여 예측 결과를 생성해주세요.")
        return None

def classify_risk(probability):
    """취소 확률에 따라 위험도를 분류합니다."""
    if probability < 0.3:
        return "안전군"
    elif probability < 0.6:
        return "주의군"
    else:
        return "위험군"

def get_risk_color(risk_level):
    """위험도에 따른 색상을 반환합니다."""
    colors = {
        "안전군": "#28a745",  # 초록색
        "주의군": "#ffc107",  # 노란색
        "위험군": "#dc3545"   # 빨간색
    }
    return colors.get(risk_level, "#6c757d")

# 메인 페이지
st.title("🏨 호텔 예약 취소 위험도 관리 대시보드")
st.markdown("---")

# 데이터 로드
df = load_predictions()

if df is not None:
    # 위험도 분류 추가
    df['risk_level'] = df['probability_cancel'].apply(classify_risk)
    df['risk_color'] = df['risk_level'].apply(get_risk_color)
    
    # 사이드바 - 필터 설정
    st.sidebar.header("⚙️ 필터 설정")
    
    # 임계값 조정
    st.sidebar.subheader("위험도 임계값 설정")
    threshold_safe = st.sidebar.slider(
        "안전군 상한 (이하)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.3, 
        step=0.05
    )
    threshold_caution = st.sidebar.slider(
        "주의군 상한 (이하)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.6, 
        step=0.05
    )
    
    # 사용자 정의 임계값으로 재분류
    def classify_risk_custom(probability):
        if probability < threshold_safe:
            return "안전군"
        elif probability < threshold_caution:
            return "주의군"
        else:
            return "위험군"
    
    df['risk_level'] = df['probability_cancel'].apply(classify_risk_custom)
    df['risk_color'] = df['risk_level'].apply(get_risk_color)
    
    # 위험도 필터
    st.sidebar.subheader("위험도 필터")
    risk_filter = st.sidebar.multiselect(
        "표시할 위험도 선택",
        options=["안전군", "주의군", "위험군"],
        default=["안전군", "주의군", "위험군"]
    )
    
    # 필터 적용
    filtered_df = df[df['risk_level'].isin(risk_filter)]
    
    # 상단 통계 카드
    col1, col2, col3, col4 = st.columns(4)
    
    total_count = len(df)
    safe_count = len(df[df['risk_level'] == "안전군"])
    caution_count = len(df[df['risk_level'] == "주의군"])
    danger_count = len(df[df['risk_level'] == "위험군"])
    
    with col1:
        st.metric(
            label="전체 예약",
            value=f"{total_count}건"
        )
    
    with col2:
        st.metric(
            label="🟢 안전군",
            value=f"{safe_count}건",
            delta=f"{safe_count/total_count*100:.1f}%"
        )
    
    with col3:
        st.metric(
            label="🟡 주의군",
            value=f"{caution_count}건",
            delta=f"{caution_count/total_count*100:.1f}%"
        )
    
    with col4:
        st.metric(
            label="🔴 위험군",
            value=f"{danger_count}건",
            delta=f"{danger_count/total_count*100:.1f}%"
        )
    
    st.markdown("---")
    
    # 차트 영역
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("📊 위험도 분포")
        
        # 파이 차트
        risk_counts = df['risk_level'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            marker=dict(colors=[get_risk_color(level) for level in risk_counts.index]),
            hole=0.4
        )])
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_chart2:
        st.subheader("📈 취소 확률 분포")
        
        # 히스토그램
        fig_hist = px.histogram(
            df,
            x='probability_cancel',
            nbins=20,
            color='risk_level',
            color_discrete_map={
                "안전군": "#28a745",
                "주의군": "#ffc107",
                "위험군": "#dc3545"
            },
            labels={'probability_cancel': '취소 확률', 'count': '고객 수'}
        )
        fig_hist.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    
    # 상세 테이블
    st.subheader("📋 고객별 위험도 상세")
    
    # 정렬 옵션
    sort_col1, sort_col2 = st.columns([3, 1])
    with sort_col1:
        sort_by = st.selectbox(
            "정렬 기준",
            options=['probability_cancel', 'risk_level', 'client_id'],
            format_func=lambda x: {
                'probability_cancel': '취소 확률 (높은 순)',
                'risk_level': '위험도',
                'client_id': '고객 ID'
            }.get(x, x)
        )
    
    with sort_col2:
        ascending = st.checkbox("오름차순", value=False)
    
    # 정렬 적용
    display_df = filtered_df.copy()
    if sort_by == 'probability_cancel':
        display_df = display_df.sort_values('probability_cancel', ascending=ascending)
    elif sort_by == 'risk_level':
        risk_order = {"위험군": 3, "주의군": 2, "안전군": 1}
        display_df['risk_order'] = display_df['risk_level'].map(risk_order)
        display_df = display_df.sort_values('risk_order', ascending=ascending)
        display_df = display_df.drop('risk_order', axis=1)
    else:
        display_df = display_df.sort_values('client_id', ascending=ascending)
    
    # 테이블 표시용 데이터 준비
    if 'client_id' in display_df.columns:
        table_df = display_df[['client_id', 'probability_cancel', 'probability_no_cancel', 'risk_level']].copy()
        table_df.columns = ['고객 ID', '취소 확률', '유지 확률', '위험도']
    else:
        table_df = display_df[['probability_cancel', 'probability_no_cancel', 'risk_level']].copy()
        table_df.columns = ['취소 확률', '유지 확률', '위험도']
    
    # 확률을 퍼센트로 변환
    table_df['취소 확률'] = table_df['취소 확률'].apply(lambda x: f"{x*100:.2f}%")
    table_df['유지 확률'] = table_df['유지 확률'].apply(lambda x: f"{x*100:.2f}%")
    
    # 스타일 적용 함수
    def highlight_risk(row):
        color = get_risk_color(row['위험도'])
        return [f'background-color: {color}20' for _ in row]
    
    styled_table = table_df.style.apply(highlight_risk, axis=1)
    
    st.dataframe(styled_table, use_container_width=True, height=400)
    
    # 다운로드 버튼
    st.markdown("---")
    col_download1, col_download2 = st.columns([1, 5])
    
    with col_download1:
        csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 CSV 다운로드",
            data=csv,
            file_name="risk_analysis.csv",
            mime="text/csv"
        )
    
    # 위험군 고객 알림
    if danger_count > 0:
        st.markdown("---")
        st.warning(f"⚠️ **주의**: {danger_count}명의 고객이 취소 위험군에 속합니다. 사전 대응이 필요합니다.")
        
        danger_customers = df[df['risk_level'] == "위험군"]
        if 'client_id' in danger_customers.columns:
            st.write("**위험군 고객 목록:**")
            danger_list = danger_customers[['client_id', 'probability_cancel']].copy()
            danger_list['probability_cancel'] = danger_list['probability_cancel'].apply(lambda x: f"{x*100:.2f}%")
            danger_list.columns = ['고객 ID', '취소 확률']
            st.dataframe(danger_list, use_container_width=True)

else:
    st.warning("예측 결과 파일이 없습니다. modeling/test.py를 먼저 실행해주세요.")
