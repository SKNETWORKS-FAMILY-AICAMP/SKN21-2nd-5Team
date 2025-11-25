import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -------------------- 1. 데이터 로드 및 전처리 --------------------
# Streamlit 캐싱을 사용하여 파일 로드 속도 최적화
@st.cache_data
def load_data(file_name="../data/hotel_bookings.csv"):
    try:
        # 이 함수 내에서 기본적인 클리닝을 수행합니다.
        df = pd.read_csv(file_name)

        # 연도 업데이트
        year_mapping = {
            2015: 2022,
            2016: 2023,
            2017: 2024
        }
        df['arrival_date_year'] = df['arrival_date_year'].replace(year_mapping)
        
        # [클리닝] ADR 음수 값 처리 및 총 투숙객 계산
        df['adr'] = df['adr'].apply(lambda x: x if x >= 0 else 0)
        df['total_guests'] = df['adults'] + df['children'] + df['babies']
        df = df[df['total_guests'] > 0].copy()
        
        # [특성 공학] 총 숙박 일수
        df['total_stays'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
        
        # [특성 공학] ADR을 사용한 간략 수익 계산 (예시)
        df['estimated_revenue'] = df['adr'] * df['total_stays'] * (1 - df['is_canceled'])
        
        # [날짜 처리] 월 순서 정렬을 위한 딕셔너리
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        df['arrival_date_month'] = pd.Categorical(df['arrival_date_month'], categories=month_order, ordered=True)
        
        # [결측치 처리]
        df['country'].fillna('Unknown', inplace=True)
        
        return df
    except FileNotFoundError:
        st.error(f"🛑 오류: '{file_name}' 파일을 찾을 수 없습니다. 파일을 확인해 주세요.")
        return pd.DataFrame()

# -------------------- 2. 대시보드 레이아웃 함수 --------------------

def run_dashboard():
    st.set_page_config(layout="wide")
    st.title("🏨 호텔 예약 통계 분석 대시보드 (관리자용)")
    
    data = load_data()
    if data.empty:
        return

    # 2-1. 사이드바 필터링
    st.sidebar.header("필터 설정")
    
    # 1. 호텔 유형 필터
    hotel_types = sorted(data['hotel'].unique())
    selected_hotels = st.sidebar.multiselect("호텔 유형 선택", hotel_types, default=hotel_types)
    
    # 2. 연도 필터
    arrival_years = sorted(data['arrival_date_year'].unique())
    selected_years = st.sidebar.multiselect("도착 연도 선택", arrival_years, default=arrival_years)
    
    # 데이터 필터링 적용
    filtered_data = data[
        (data['hotel'].isin(selected_hotels)) & 
        (data['arrival_date_year'].isin(selected_years))
    ]

    if filtered_data.empty:
        st.warning("선택된 필터에 해당하는 데이터가 없습니다. 필터를 조정해 주세요.")
        return

    # 2-2. KPI 요약 카드
    st.header("🔑 핵심 성과 지표 (KPIs)")
    
    # KPIs 계산
    total_bookings = len(filtered_data)
    total_cancellations = filtered_data['is_canceled'].sum()
    cancellation_rate = total_cancellations / total_bookings
    avg_adr = filtered_data['adr'].mean()
    avg_stays = filtered_data['total_stays'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 예약 건수", f"{total_bookings:,} 건")
    with col2:
        st.metric("순 취소율", f"{cancellation_rate:.2%}")
    with col3:
        st.metric("평균 ADR", f"$ {avg_adr:,.0f}")
    with col4:
        st.metric("평균 숙박 일수", f"{avg_stays:.1f} 일")
        
    st.markdown("---")
        
    # 2-3. Section A: 취소 및 리스크 관리 분석
    st.header("🚨 Section A: 취소 및 리스크 관리")
    
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        st.subheader("A-1. 월별 취소율 추이")
        cancel_trend = filtered_data.groupby('arrival_date_month')['is_canceled'].mean().reset_index()
        cancel_trend.columns = ['Month', 'Cancellation Rate']
        fig_a1 = px.line(cancel_trend, x='Month', y='Cancellation Rate', 
                         title='월별 예약 취소율', markers=True, 
                         height=400)
        fig_a1.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig_a1, use_container_width=True)

    with col_a2:
        st.subheader("A-2. 취소율 vs. 보증금 유형 및 리드타임")
        risk_deposit = filtered_data.groupby(['deposit_type', 'lead_time'])['is_canceled'].mean().reset_index()
        # 리드 타임 5분위수 그룹 생성 (시각화를 위해)
        risk_deposit['lead_time_group'] = pd.qcut(risk_deposit['lead_time'], q=5, 
                                                labels=[f'Q{i}' for i in range(1, 6)], 
                                                duplicates='drop')
        
        risk_analysis = risk_deposit.groupby(['deposit_type', 'lead_time_group'])['is_canceled'].mean().reset_index()
        risk_analysis.columns = ['Deposit Type', 'Lead Time Group', 'Cancellation Rate']
        
        fig_a2 = px.bar(risk_analysis, x='Deposit Type', y='Cancellation Rate', 
                        color='Lead Time Group', barmode='group',
                        title='보증금 및 선행기간별 취소 위험도', height=400)
        fig_a2.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig_a2, use_container_width=True)

    # 2-4. Section B: 재무 및 수익 분석
    st.markdown("---")
    st.header("💰 Section B: 재무 및 수익 분석")
    
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        st.subheader("B-1. 월별 평균 ADR 추이 (호텔 비교)")
        adr_trend = filtered_data.groupby(['arrival_date_month', 'hotel'])['adr'].mean().reset_index()
        adr_trend.columns = ['Month', 'Hotel Type', 'Average ADR']
        fig_b1 = px.line(adr_trend, x='Month', y='Average ADR', color='Hotel Type',
                         title='호텔 유형별 월 평균 ADR', markers=True, height=400)
        st.plotly_chart(fig_b1, use_container_width=True)

    with col_b2:
        st.subheader("B-2. 시장 채널별 수익 기여도")
        # 예상 순수익 (취소되지 않은 예약의 ADR * Total Stays)
        revenue_segment = filtered_data.groupby('market_segment')['estimated_revenue'].sum().reset_index()
        revenue_segment.columns = ['Market Segment', 'Total Estimated Revenue']
        revenue_segment = revenue_segment.sort_values('Total Estimated Revenue', ascending=False)
        
        fig_b2 = px.bar(revenue_segment.head(10), x='Market Segment', y='Total Estimated Revenue',
                        title='상위 10개 시장 채널별 총 수익 기여', height=400)
        st.plotly_chart(fig_b2, use_container_width=True)

    # 2-5. Section C: 운영 효율 및 자원 관리
    st.markdown("---")
    st.header("⚙️ Section C: 운영 효율 및 자원 관리")
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.subheader("C-1. 예약 선행 기간(Lead Time) 분포")
        fig_c1 = px.histogram(filtered_data, x='lead_time', nbins=50, 
                             title='예약 선행 기간 분포 (일)', height=400)
        fig_c1.update_xaxes(title_text="Lead Time (Days)")
        st.plotly_chart(fig_c1, use_container_width=True)
        
    with col_c2:
        st.subheader("C-2. 객실 타입 사용 현황")
        room_usage = filtered_data.groupby(['hotel', 'reserved_room_type']).size().reset_index(name='Count')
        fig_c2 = px.bar(room_usage, x='hotel', y='Count', color='reserved_room_type',
                        title='호텔 유형별 예약된 객실 타입 비율', height=400)
        st.plotly_chart(fig_c2, use_container_width=True)

    # 2-6. Section D: 상세 데이터 테이블
    st.markdown("---")
    st.header("D. 필터 적용 상세 데이터")
    st.dataframe(filtered_data.head(1000).drop(columns=['total_guests', 'total_stays']), use_container_width=True)


if __name__ == "__main__":
    run_dashboard()