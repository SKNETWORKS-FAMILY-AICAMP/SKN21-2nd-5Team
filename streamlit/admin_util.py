import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# st.space를 대체할 헬퍼 함수 정의
def _add_vertical_space(pixels):
    st.markdown(f"<div style='height: {pixels}px;'></div>", unsafe_allow_html=True)

def visualize_customer_data(customer_row, customer_index):
    """선택된 고객 데이터 시각화하는 함수"""
    st.subheader(f"🔍 고객 정보")
    _add_vertical_space(25)

# 기본 정보
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("고객 ID", customer_row['name'])
    with col2:
        st.metric("호텔 타입", customer_row['hotel'])
        _add_vertical_space(25)
        st.metric("총 인원", f"{int(customer_row['adults']) + int(customer_row['children']) + int(customer_row['babies'])}")
        st.write(f"(성인 {int(customer_row['adults'])}, 어린이 {int(customer_row['children'])}, 유아 {int(customer_row['babies'])}명)")
        _add_vertical_space(25)
        st.metric("유통 채널", f"{customer_row['distribution_channel']}")
        _add_vertical_space(15)
        st.metric("조식", customer_row['meal'])
    with col3:
        st.metric("방문예정일", f"{int(customer_row['arrival_date_year'])}-{int(customer_row['arrival_date_month']):02d}-{int(customer_row['arrival_date_day_of_month']):02d}")
        _add_vertical_space(25)
        st.metric("요금", f"${(customer_row['adr']) * (int(customer_row['stays_in_weekend_nights']) + int(customer_row['stays_in_week_nights'])):.2f}")
        st.write(f"(${customer_row['adr']:.2f} x {int(customer_row['stays_in_weekend_nights']) + int(customer_row['stays_in_week_nights'])}박)")
        _add_vertical_space(25)
        st.metric("보증금", f"{customer_row['deposit_type']}")
        _add_vertical_space(15)
        st.metric("주차 공간", customer_row['required_car_parking_spaces'])
    st.markdown("---")
    
    if customer_row['total_of_special_requests'] > 0:
        special_requests = customer_row['customer_special_requests']
    if special_requests and special_requests != 'None':
        requests_list = [req.strip().strip("'") for req in special_requests.split('/') if req.strip()]
        st.write(f"**요청사항 ({customer_row['total_of_special_requests']}건)**")
        for request in requests_list:
            st.write(f"- {request}")
    else:
        st.write("**특별 요청사항:** 없음")
    st.markdown("---")

# 위험도 분류 및 색상 함수
def classify_risk(prob):
    """취소 확률에 따른 위험도 분류 함수"""
    if prob < 0.3:
        return "🟢 안전군", "green"
    elif prob < 0.7:
        return "🟡 주의군", "orange"
    else:
        return "🔴 위험군", "red"

# 고객 예측 결과 추출 함수
def get_customer_prediction(customer_row, predictions_df):
    """고객의 취소 예측 결과를 추출하는 함수"""
    customer_id = customer_row['name']
    prediction_row = predictions_df[predictions_df['name'] == customer_id]

    if prediction_row.empty:
        return None

    prediction = int(prediction_row.iloc[0]['prediction'])
    prob_no_cancel = float(prediction_row.iloc[0]['probability_no_cancel'])
    prob_cancel = float(prediction_row.iloc[0]['probability_cancel'])

    return {
        'customer_id': customer_id,
        'prediction': prediction,
        'prob_no_cancel': prob_no_cancel,
        'prob_cancel': prob_cancel
    }

# 예측 결과 시각화 함수
def display_prediction_results(prediction_data, predictions_df):
    """예측 결과를 시각화하는 함수"""
    st.subheader(f"⌛️ 예약 취소 예측")
    _add_vertical_space(25)
    if prediction_data is None:
        st.error(f"❌ 고객의 예측 결과를 찾을 수 없습니다.")
        return

    prediction = prediction_data['prediction']
    prob_no_cancel = prediction_data['prob_no_cancel']
    prob_cancel = prediction_data['prob_cancel']

    # 위험도 분류
    risk_level, risk_color = classify_risk(prob_cancel)

    # 메인 메트릭 표시
    st.markdown(f"#### {risk_level}")
    _add_vertical_space(25) # st.space(size=5) 대체 (5 * 5px = 25px)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="📊 취소 예측", value="취소" if prediction == 1 else "유지")
    with col2:
        st.metric(label="🔴 **취소** 확률", value=f"{prob_cancel*100:.2f}%")
    with col3:
        st.metric(label="🟢 **유지** 확률", value=f"{prob_no_cancel*100:.2f}%")

    col4, col5 = st.columns(2)
    with col4:
        # 원형 차트
        fig_bar = go.Figure(data=[
            go.Pie(labels=['유지 확률', '취소 확률'],
                   values=[prob_no_cancel*100,prob_cancel*100],
                   marker_colors=['lightblue', 'lightcoral'],
                   textposition='outside',
                   textinfo='percent+label')
                   ])
        fig_bar.update_layout(
            title="예측 확률 비교",
            yaxis_title="확률 (%)",
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    with col5:
        # 게이지 차트
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob_cancel * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {'axis': {'range': [None, 100]},
                     'bar': {'color': risk_color},
                     'steps': [{'range': [0, 30], 'color': "#90EE90"},
                               {'range': [30, 70], 'color': "#FFD700"},
                               {'range': [70, 100], 'color': "#F08080"}
                               ],
                     'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}
                    }))

        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown("---")

    # 추천 액션
    col1, spacer, col2 = st.columns([3, 0.5, 2])

    with col1:
        st.markdown("#### 💡 추천 액션")
        _add_vertical_space(25) # st.space(size=5) 대체 (5 * 5px = 25px)
        if prob_cancel < 0.3:
            st.success("✅ **안전한 예약**")
            st.write("- 특별 조치 필요 없음")
            st.write("- 고객 만족도 관리 유지")
        elif prob_cancel < 0.7:
            st.warning("⚠️ **케어 필요 예약**")
            st.write("- 고객 만족도 향상을 위한 추가 서비스 제공 고려")
            st.write("- 예약 확인 연락 및 특별 혜택 제공 검토")
            st.write("- 고객의 특별 요청사항 적극 수용")
        else:
            st.error("🚨 **취소 위험 높은 예약**")
            st.write("- 고객 관리팀의 개입이 필요")
            st.write("- 개인화된 서비스 제공 및 할인 혜택 고려")
            st.write("- 예약 재확인 및 고객 니즈 파악을 위한 직접 연락 권장")
            st.write("- 룸 업그레이드 또는 추가 어메니티 제공 검토")

    # 유사 고객 비교
    with col2:
        st.markdown("#### 📊 유사 고객 비교")
        _add_vertical_space(25) # st.space(size=5) 대체 (5 * 5px = 25px)
        if not predictions_df.empty and 'probability_cancel' in predictions_df.columns:
            avg_cancel_prob = predictions_df['probability_cancel'].mean()
            if len(predictions_df) > 1:
                percentile_rank = (predictions_df['probability_cancel'] < prob_cancel).sum() / (len(predictions_df) - 1) * 100 if (len(predictions_df) -1) > 0 else 0
            else:
                percentile_rank = 100 if prob_cancel > 0 else 0

            st.metric(
                "전체 평균 취소 확률",
                f"{avg_cancel_prob*100:.2f}%",
                f"{(prob_cancel - avg_cancel_prob)*100:+.2f}%",
                delta_color="inverse"
            )
            st.metric(
                "위험도 순위",
                f"하위 {percentile_rank:.1f}%",
                delta_color="inverse"
            )
        else:
            st.warning("예측 데이터프레임이 비어있거나 'probability_cancel' 컬럼이 없습니다.")
