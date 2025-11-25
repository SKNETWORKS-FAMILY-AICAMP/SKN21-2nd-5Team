import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from streamlit_elements import elements, mui, html, dashboard, editor, nivo, media, sync, lazy, event 

# /Users/wjsdndud/SKNAIcamp/02_Project/SKN21-2nd-5Team/streamlit/admin_personal_customer.py

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'hotel_bookings_data.csv')
PREDICTIONS_PATH = os.path.join(BASE_DIR, 'data', 'test_predictions.csv')

customer_df = pd.read_csv(DATA_PATH)
predictions_df = pd.read_csv(PREDICTIONS_PATH)

# 파일 존재 여부 확인
if not os.path.exists(DATA_PATH):
    st.error(f"❌ 파일을 찾을 수 없습니다: {DATA_PATH}")
    st.stop()

with st.sidebar:
    st.title("🔑 관리자 메뉴")
    st.page_link(page = "http://localhost:8501/", label = "메인 통계", icon="📊", )
    st.page_link(page = "http://localhost:8501/", label = "개별 고객 정보 분석", icon="👤")


def visualize_customer_data(customer_row, customer_index):
    """선택된 고객 데이터 시각화하는 함수"""
    
# 기본 정보
    st.markdown(f"**기본 정보**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("고객 ID", customer_row['name'])
        st.metric("총 인원", f"{int(customer_row['adults']) + int(customer_row['children']) + int(customer_row['babies'])}") 
        st.write(f"(성인: {int(customer_row['adults'])}, 어린이: {int(customer_row['children'])}, 유아: {int(customer_row['babies'])})")
    with col2:
        st.metric("호텔 타입", customer_row['hotel'])
        st.metric("투숙일까지", f"D-{customer_row['lead_time']}")
    with col3:
        # 날짜 컬럼들을 직접 f-string으로 형식화
        st.metric("방문예정일", f"{int(customer_row['arrival_date_year'])}-{int(customer_row['arrival_date_month']):02d}-{int(customer_row['arrival_date_day_of_month']):02d}")
        # st.metric("룸 타입", customer_row['reserved_room_type'])
        st.metric("요금", f"${(customer_row['adr']) * (int(customer_row['stays_in_weekend_nights']) + int(customer_row['stays_in_week_nights'])):.2f}")
        st.write(f"(${customer_row['adr']:.2f} x {int(customer_row['stays_in_weekend_nights']) + int(customer_row['stays_in_week_nights'])}박)")
    st.markdown("---")
    st.markdown(f"**부가 정보**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Market Segment**: {customer_row['market_segment']}")
    with col2:
        st.write(f"**Distribution Channel**: {customer_row['distribution_channel']}")
    with col3:
        st.write(f"**Deposit Type**: {customer_row['deposit_type']}")
    st.markdown("---")
# 고객 요청사항
    st.markdown(f"**고객 요청사항**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("조식", customer_row['meal'])
    with col2:
        st.metric("주차 공간", customer_row['required_car_parking_spaces'])
    with col3:
        st.metric("특별 요청", f"{customer_row['total_of_special_requests']}개")
    st.markdown("<br>", unsafe_allow_html=True)
    if customer_row['total_of_special_requests'] > 0:
        st.write("- 특별 요청 사항 내용 표시")
    st.markdown("---")
    

    
    # 예측 결과 추가
    st.markdown("---")
    predict_customer_cancel(customer_row, customer_index)
    
def predict_customer_cancel(customer_row, customer_index):
    """선택된 고객의 취소 예측 결과를 시각화하는 함수"""
    
    # 고객 ID로 예측 결과 찾기
    customer_id = customer_row['name']
    prediction_row = predictions_df[predictions_df['client_id'] == customer_id]
    
    if prediction_row.empty:
        st.error(f"❌ {customer_id} 고객의 예측 결과를 찾을 수 없습니다.")
        return
    
    # 예측 결과 추출
    prediction = int(prediction_row.iloc[0]['prediction'])
    prob_no_cancel = prediction_row.iloc[0]['probability_no_cancel']
    prob_cancel = prediction_row.iloc[0]['probability_cancel']
    
    # 메인 메트릭 표시
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🎯 취소 예측",
            value="취소" if prediction == 1 else "유지",
            delta="위험" if prediction == 1 else "안전"
        )
    
    with col2:
        st.metric(
            label="📈 취소 확률",
            value=f"{prob_cancel*100:.2f}%",
            delta=f"{(prob_cancel - 0.5)*100:.2f}%"
        )
    
    with col3:
        st.metric(
            label="📉 유지 확률",
            value=f"{prob_no_cancel*100:.2f}%",
            delta=f"{(prob_no_cancel - 0.5)*100:.2f}%"
        )
    
    # 위험도 분류 및 색상 함수
    def classify_risk(prob):
        if prob < 0.3:
            return "🟢 안전군", "green"
        elif prob < 0.7:
            return "🟡 주의군", "orange"
        else:
            return "🔴 위험군", "red"
    
    risk_level, risk_color = classify_risk(prob_cancel)
    
    # 위험도 표시
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 위험도 분류: {risk_level}")
        
        # 확률 바 차트
        fig_bar = go.Figure(data=[
            go.Bar(
                x=['유지 확률', '취소 확률'],
                y=[prob_no_cancel*100, prob_cancel*100],
                marker_color=['lightblue', 'lightcoral'],
                text=[f"{prob_no_cancel*100:.1f}%", f"{prob_cancel*100:.1f}%"],
                textposition='auto'
            )
        ])
        
        fig_bar.update_layout(
            title="예측 확률 비교",
            yaxis_title="확률 (%)",
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # 게이지 차트
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prob_cancel * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "취소 위험도"},
            delta = {'reference': 50},
            gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': risk_color},
            'steps': [
                {'range': [0, 30], 'color': "#90EE90"},
                {'range': [30, 70], 'color': "#FFD700"},
                {'range': [70, 100], 'color': "#F08080"}
            ],
            'threshold': {
                'line': {'color': "#FF0000", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
            }
        ))
        
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # 추천 액션
    st.markdown("---")
    st.subheader("💡 추천 액션")
    
    if prob_cancel < 0.3:
        st.success("✅ **안전한 예약입니다**")
        st.write("- 특별한 조치가 필요하지 않습니다.")
        st.write("- 정기적인 고객 만족도 관리를 유지하세요.")
    elif prob_cancel < 0.7:
        st.warning("⚠️ **주의가 필요한 예약입니다**")
        st.write("- 고객 만족도 향상을 위한 추가 서비스 제공을 고려하세요.")
        st.write("- 예약 확인 연락 및 특별 혜택 제공을 검토하세요.")
        st.write("- 고객의 특별 요청사항을 적극적으로 수용하세요.")
    else:
        st.error("🚨 **취소 위험이 높은 예약입니다**")
        st.write("- 즉시 고객 관리팀의 개입이 필요합니다.")
        st.write("- 개인화된 서비스 제공 및 할인 혜택을 고려하세요.")
        st.write("- 예약 재확인 및 고객 니즈 파악을 위한 직접 연락을 권장합니다.")
        st.write("- 룸 업그레이드나 추가 어메니티 제공을 검토하세요.")
    
    # 유사한 고객 패턴 분석
    st.markdown("---")
    st.subheader("📊 유사한 고객 패턴 분석")
    
    # 전체 데이터에서 취소 확률 분포
    avg_cancel_prob = predictions_df['probability_cancel'].mean()
    percentile_rank = (predictions_df['probability_cancel'] <= prob_cancel).mean() * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "전체 평균 취소 확률",
            f"{avg_cancel_prob*100:.2f}%",
            f"{(prob_cancel - avg_cancel_prob)*100:+.2f}%"
        )
    
    with col2:
        st.metric(
            "위험도 순위",
            f"상위 {100-percentile_rank:.1f}%",
            f"{'높음' if percentile_rank > 70 else '보통' if percentile_rank > 30 else '낮음'}"
        )

 
st.title("🏨 개별 고객 정보 분석 시스템")
st.markdown("---")

# 고객 선택 방법
st.subheader("🎯 고객 선택 방법")

selection_method = st.radio(
    "고객을 선택하는 방법을 선택하세요:",
    ["테이블에서 행 선택", "고객 ID로 직접 검색"]
)

if selection_method == "고객 ID로 직접 검색":
    # 고객 ID 직접 입력
    customer_ids = customer_df['name'].unique() if 'name' in customer_df.columns else []
    selected_customer_id = st.selectbox(
        "분석할 고객 ID를 선택하세요:",
        options=customer_ids
    )
    
    if selected_customer_id:
        selected_row = customer_df[customer_df['name'] == selected_customer_id].iloc[0]
        selected_index = customer_df[customer_df['name'] == selected_customer_id].index[0]
        visualize_customer_data(selected_row, selected_index)

else:
    # 데이터 테이블 표시
    st.subheader("📋 고객 데이터 테이블")
    st.write("⬇️ 아래 테이블에서 행을 선택하여 상세 분석을 확인하세요")
    
    # 인터랙티브 데이터프레임 (행 선택 가능)
    events = st.dataframe(
        data=customer_df,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    # 행이 선택되었을 때 시각화
    if events.selection["rows"]:
        selected_indices = events.selection["rows"]
        selected_index = selected_indices[0]
        selected_row = customer_df.iloc[selected_index]
        
        st.markdown("---")
        visualize_customer_data(selected_row, selected_index)

