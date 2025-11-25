import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------- 1. 데이터 로드 및 전처리 --------------------
@st.cache_data
def load_data(file_name="data/hotel_bookings.csv"):
    try:
        df = pd.read_csv(file_name)

        # 연도 업데이트
        year_mapping = {2015: 2022, 2016: 2023, 2017: 2024}
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
        st.error(f"🛑 오류: '파일 이름' 파일을 찾을 수 없습니다. 파일을 확인해 주세요.")
        return pd.DataFrame()

# -------------------- 2. 공통 차트 생성 함수 (함수화하여 코드 중복 제거) --------------------

def generate_charts(filtered_data, dashboard_title):
    # filtered_data에 'lead_time_group' 파생 변수 생성 (A-2에서 사용)
    # qcut이 오류나지 않도록 최소 데이터 건수 확인 (필터링 결과가 적을 경우 오류 방지)
    if len(filtered_data['lead_time'].dropna().unique()) >= 5:
        try:
            filtered_data['lead_time_group'] = pd.qcut(filtered_data['lead_time'], q=5, 
                                                       labels=[f'Q{i}' for i in range(1, 6)], 
                                                       duplicates='drop')
        except ValueError:
             filtered_data['lead_time_group'] = 'Group_1' # 데이터가 충분치 않으면 단일 그룹 지정
    else:
        filtered_data['lead_time_group'] = 'Group_1'


    st.title(dashboard_title)

    if filtered_data.empty:
        st.warning("선택된 필터에 해당하는 데이터가 없습니다. 필터를 조정해 주세요.")
        return

    # 2-2. KPI 요약 카드
    st.header("🔑 핵심 성과 지표 (KPIs)")
    
    total_bookings = len(filtered_data)
    total_cancellations = filtered_data['is_canceled'].sum()
    cancellation_rate = total_cancellations / total_bookings if total_bookings > 0 else 0
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
        
        # 호텔 유형이 하나만 선택된 경우에도 동일 구조 유지 (City Hotel 전용 페이지)
        if filtered_data['hotel'].nunique() > 1:
            hotel_cancel_trend = filtered_data.groupby(['arrival_date_month', 'hotel'])['is_canceled'].mean().reset_index()
            hotel_cancel_trend.columns = ['Month', 'Hotel Type', 'Cancellation Rate']
            overall_cancel_trend = filtered_data.groupby('arrival_date_month')['is_canceled'].mean().reset_index()
            overall_cancel_trend['Hotel Type'] = '통합 전체'
            overall_cancel_trend.columns = ['Month', 'Cancellation Rate', 'Hotel Type']
            combined_cancel_trend = pd.concat([hotel_cancel_trend, overall_cancel_trend])
        else:
            combined_cancel_trend = filtered_data.groupby('arrival_date_month')['is_canceled'].mean().reset_index()
            combined_cancel_trend.columns = ['Month', 'Cancellation Rate']
            combined_cancel_trend['Hotel Type'] = filtered_data['hotel'].iloc[0]

        fig_a1 = px.line(combined_cancel_trend, x='Month', y='Cancellation Rate', 
                         color='Hotel Type', title='월별 예약 취소율', markers=True, 
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         height=400)
        fig_a1.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig_a1, use_container_width=True)

    with col_a2:
        st.subheader("A-2. 취소율 vs. 보증금 유형 및 리드타임 그룹")
        
        risk_analysis = filtered_data.groupby(['deposit_type', 'lead_time_group'])['is_canceled'].mean().reset_index()
        risk_analysis.columns = ['Deposit Type', 'Lead Time Group', 'Cancellation Rate']
        
        fig_a2 = px.bar(risk_analysis, x='Deposit Type', y='Cancellation Rate', 
                         color='Lead Time Group', barmode='group',
                         title='보증금 및 선행기간 그룹별 취소 위험도', 
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         height=400)
        fig_a2.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig_a2, use_container_width=True)
    
    st.subheader("A-3. 리드타임이 취소에 미치는 영향")

    # [한글 폰트 설정]
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
    except:
        plt.rcParams['font.family'] = 'sans-serif' 
        
    plt.rcParams['axes.unicode_minus'] = False 
    
    # 1. 데이터 필터링: 극단적인 이상치 제거 및 Plotting을 위한 데이터 준비
    lt_max = filtered_data['lead_time'].quantile(0.99)
    plot_data_density = filtered_data[filtered_data['lead_time'] <= lt_max].copy() 
    
    plot_data_density['is_canceled_label'] = plot_data_density['is_canceled'].map({0: '미취소 (0)', 1: '취소 (1)'})
    
    # 2. Seaborn KDE Plot 생성
    fig, ax = plt.subplots(figsize=(10, 5))
    pastel_colors = {'미취소 (0)': '#85c793', '취소 (1)': '#ffba49'}
    
    sns.kdeplot(data=plot_data_density, x='lead_time', hue='is_canceled_label',
                hue_order=['미취소 (0)', '취소 (1)'],
                palette=pastel_colors,
                fill=True, alpha=.7, linewidth=2, ax=ax)
    
    ax.set_title('취소 여부별 예약 선행 기간(Lead Time) 분포 밀도')
    ax.set_xlabel('예약 선행 기간 (Lead Time)')
    ax.set_ylabel('밀도')
    plt.tight_layout()
    
    st.pyplot(fig)
    plt.close(fig) 
    
    # 2-4. Section B: 재무 및 수익 분석
    st.markdown("---")
    st.header("💰 Section B: 재무 및 수익 분석")
    
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        st.subheader("B-1. 월별 평균 ADR 추이")
        
        adr_trend = filtered_data.groupby(['arrival_date_month', 'hotel'])['adr'].mean().reset_index()
        adr_trend.columns = ['Month', 'Hotel Type', 'Average ADR']
        fig_b1 = px.line(adr_trend, x='Month', y='Average ADR', color='Hotel Type',
                         title='호텔 유형별 월 평균 ADR', markers=True, 
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         height=400)
        st.plotly_chart(fig_b1, use_container_width=True)

    with col_b2:
        st.subheader("B-2. 시장 채널별 수익 기여도")
        
        revenue_segment = filtered_data.groupby('market_segment')['estimated_revenue'].sum().reset_index()
        revenue_segment.columns = ['Market Segment', 'Total Estimated Revenue']
        revenue_segment = revenue_segment.sort_values('Total Estimated Revenue', ascending=False)
        
        fig_b2 = px.bar(revenue_segment.head(10), x='Market Segment', y='Total Estimated Revenue',
                         title='상위 10개 시장 채널별 총 수익 기여', 
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         height=400)
        st.plotly_chart(fig_b2, use_container_width=True)

    # 2-5. Section C: 운영 효율 및 자원 관리
    st.markdown("---")
    st.header("⚙️ Section C: 운영 효율 및 자원 관리")
    
    col_c1, col_c2 = st.columns(2)

    # C-1. 변경 예약 건수 분포 (운영 부하 측정)
    with col_c1:
        st.subheader("C-1. 변경 예약 건수 분포")
        
        changes_counts = filtered_data['booking_changes'].value_counts().reset_index()
        changes_counts.columns = ['Changes Count', 'Count']
        
        top_n = 5
        other_count = changes_counts[changes_counts['Changes Count'] >= top_n]['Count'].sum()
        
        changes_plot = changes_counts[changes_counts['Changes Count'] < top_n].copy()
        changes_plot.loc[len(changes_plot)] = [f'{top_n}+ 이상', other_count]
        
        fig_c1 = px.bar(changes_plot, x='Changes Count', y='Count',
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         title='예약 변경 건수 빈도 (프론트 부하)')
        st.plotly_chart(fig_c1, use_container_width=True)


    # C-2. 총 요청 건수 분포 (자원 소모 측정)
    with col_c2:
        st.subheader("C-2. 특별 요청 건수 분포")
        
        fig_c2 = px.histogram(filtered_data, x='total_of_special_requests',
                              color='is_canceled', barmode='group',
                              color_discrete_sequence=px.colors.qualitative.Pastel,
                              title='총 특별 요청 건수 빈도 (하우스키핑/컨시어지 부하)')
        fig_c2.update_xaxes(categoryorder='total ascending')
        st.plotly_chart(fig_c2, use_container_width=True)

    st.markdown("---")
    col_c3, col_c4 = st.columns(2)

    # C-3. 주말/주중 숙박 비율 (인력 수요 패턴 분석)
    with col_c3:
        st.subheader("C-3. 주말 vs. 주중 숙박 비율 분포")
        
        stays_data = filtered_data[['stays_in_weekend_nights', 'stays_in_week_nights', 'hotel']].copy()
        stays_data_melt = stays_data.melt(
            id_vars=['hotel'], 
            value_vars=['stays_in_weekend_nights', 'stays_in_week_nights'],
            var_name='Stay Type', 
            value_name='Nights'
        )
        stays_data_melt = stays_data_melt[stays_data_melt['Nights'] > 0]
        
        fig_c3 = px.box(
            stays_data_melt, 
            x='Stay Type', 
            y='Nights', 
            color='hotel', 
            points="outliers",
            title='호텔 유형별 주말/주중 숙박 일수 분포',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            height=450
        )
        max_nights_95 = stays_data_melt['Nights'].quantile(0.95)
        fig_c3.update_yaxes(range=[0, max_nights_95]) 
        
        st.plotly_chart(fig_c3, use_container_width=True)

    # C-4. 객실 타입 사용 현황 (복구 및 유지)
    with col_c4:
        st.subheader("C-4. 객실 타입 사용 현황 (상위 4개 + 기타)")
        
        top_4_rooms = filtered_data['reserved_room_type'].value_counts().nlargest(4).index.tolist()
        
        data_for_pie = filtered_data[['reserved_room_type']].copy()
        data_for_pie['room_group'] = np.where(
            data_for_pie['reserved_room_type'].isin(top_4_rooms),
            data_for_pie['reserved_room_type'],
            '기타 (Other)'
        )
        
        room_counts = data_for_pie['room_group'].value_counts().reset_index()
        room_counts.columns = ['Reserved Room Group', 'Count']
        
        fig_c4 = px.pie(room_counts, 
                         names='Reserved Room Group', 
                         values='Count', 
                         title='전체 예약 건수 대비 객실 타입별 점유율',
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         height=450)
        
        fig_c4.update_traces(textinfo='percent+label')
        
        st.plotly_chart(fig_c4, use_container_width=True)

    # 2-6. Section D: 상세 데이터 테이블
    st.markdown("---")
    st.header("D. 필터 적용 상세 데이터")
    st.dataframe(filtered_data.head(1000).drop(columns=['total_guests', 'total_stays', 'lead_time_group'], errors='ignore'), use_container_width=True)

# -------------------- 3. 호텔 통합 분석 페이지 (기존 run_dashboard) --------------------

def run_full_dashboard(data):
    st.sidebar.header("통합 분석 필터 (City + Resort)")
    
    # 1. 호텔 유형 필터 (통합 분석에서만 필요)
    hotel_types = sorted(data['hotel'].unique())
    selected_hotels = st.sidebar.multiselect("호텔 유형 선택", hotel_types, default=hotel_types)
    
    # 2. 연도 필터
    arrival_years = sorted(data['arrival_date_year'].unique())
    selected_years = st.sidebar.multiselect("도착 연도 선택", arrival_years, default=arrival_years)
    
    # 데이터 필터링 적용
    filtered_data = data[
        (data['hotel'].isin(selected_hotels)) & 
        (data['arrival_date_year'].isin(selected_years))
    ].copy()

    generate_charts(filtered_data, "🏨 호텔 예약 통계 분석 대시보드 (통합)")

# -------------------- 4. City Hotel 전용 분석 페이지 (새로운 함수) --------------------

def run_city_hotel_dashboard(data):
    st.sidebar.header("City Hotel 전용 필터")
    
    # City Hotel만 초기 필터링
    city_data = data[data['hotel'] == 'City Hotel'].copy()

    # 1. 연도 필터
    arrival_years = sorted(city_data['arrival_date_year'].unique())
    selected_years = st.sidebar.multiselect("도착 연도 선택", arrival_years, default=arrival_years)
    
    # 2. 월 필터 (새로 추가됨)
    month_order = city_data['arrival_date_month'].cat.categories.tolist()
    selected_months = st.sidebar.multiselect("도착 월 선택", month_order, default=month_order)
    
    # 데이터 필터링 적용
    filtered_data = city_data[
        (city_data['arrival_date_year'].isin(selected_years)) &
        (city_data['arrival_date_month'].isin(selected_months))
    ].copy()

    generate_charts(filtered_data, "🏢 City Hotel 전용 분석 대시보드")

# -------------------- 5. 메인 실행 함수 (대시보드 선택 스위치) --------------------

def main():
    st.set_page_config(layout="wide")
    
    # Plotly 전역 템플릿 설정 (반복되는 코드 제거)
    pio.templates.default = "plotly_white"
    
    data = load_data()
    if data.empty:
        return

    st.sidebar.title("페이지 선택")
    dashboard_selection = st.sidebar.radio(
        "분석 페이지 선택", 
        ['통합 분석 (City + Resort)', 'City Hotel 전용 분석']
    )
    
    # 선택된 페이지에 따라 함수 실행
    if dashboard_selection == '통합 분석 (City + Resort)':
        run_full_dashboard(data)
    else:
        run_city_hotel_dashboard(data)

if __name__ == "__main__":
    main()