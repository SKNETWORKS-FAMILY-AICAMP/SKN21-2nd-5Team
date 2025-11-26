import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys

# --- 경로 설정 및 sys.path 추가 ---
current_pages_dir = os.path.dirname(os.path.abspath(__file__))
streamlit_app_dir = os.path.dirname(current_pages_dir)
project_root_dir = os.path.dirname(streamlit_app_dir)

# -------------------- 1. 데이터 로드 및 전처리 (dashboard_admin.py에서 가져옴) --------------------
@st.cache_data
def load_dashboard_data(file_name="hotel_bookings.csv"):
    try:
        file_path = os.path.join(project_root_dir, 'data', file_name)
        df = pd.read_csv(file_path)
        year_mapping = {2015: 2022, 2016: 2023, 2017: 2024}
        df['arrival_date_year'] = df['arrival_date_year'].replace(year_mapping)
        df['adr'] = df['adr'].apply(lambda x: x if x >= 0 else 0)
        df['total_guests'] = df['adults'] + df['children'] + df['babies']
        df = df[df['total_guests'] > 0].copy()
        df['total_stays'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
        df['estimated_revenue'] = df['adr'] * df['total_stays'] * (1 - df['is_canceled'])
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        df['arrival_date_month'] = pd.Categorical(df['arrival_date_month'], categories=month_order, ordered=True)
        df['country'].fillna('Unknown', inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"🛑 오류: '{file_path}' 파일을 찾을 수 없습니다. 파일을 확인해 주세요.")
        return pd.DataFrame()

# --- 개별 고객 데이터 로드 ---
DATA_PATH = os.path.join(project_root_dir, 'data', 'hotel_bookings_data.csv')
PREDICTIONS_PATH = os.path.join(project_root_dir, 'data', 'test_predictions.csv')

customer_df = pd.DataFrame()
predictions_df = pd.DataFrame()
if st.session_state.get('logged_in', False):
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ 고객 데이터를 찾을 수 없습니다: {DATA_PATH}")
    elif not os.path.exists(PREDICTIONS_PATH):
        st.error(f"❌ 예측 데이터를 찾을 수 없습니다: {PREDICTIONS_PATH}")
    else:
        customer_df = pd.read_csv(DATA_PATH)
        predictions_df = pd.read_csv(PREDICTIONS_PATH)

# --- 대시보드 데이터 로드 ---
dashboard_data = load_dashboard_data()


# -------------------- 2. 공통 차트 생성 함수 (dashboard_admin.py에서 가져옴) --------------------
def generate_charts(filtered_data, dashboard_title):
    if len(filtered_data['lead_time'].dropna().unique()) >= 5:
        try:
            filtered_data['lead_time_group'] = pd.qcut(filtered_data['lead_time'], q=5,
                                                       labels=[f'Q{i}' for i in range(1, 6)],
                                                       duplicates='drop')
        except ValueError:
             filtered_data['lead_time_group'] = 'Group_1'
    else:
        filtered_data['lead_time_group'] = 'Group_1'

    st.title(dashboard_title)

    if filtered_data.empty:
        st.warning("선택된 필터에 해당하는 데이터가 없습니다. 필터를 조정해 주세요.")
        return

    st.header("🔑 핵심 성과 지표 (KPIs)")
    total_bookings = len(filtered_data)
    total_cancellations = filtered_data['is_canceled'].sum()
    cancellation_rate = total_cancellations / total_bookings if total_bookings > 0 else 0
    avg_adr = filtered_data['adr'].mean()
    avg_stays = filtered_data['total_stays'].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("총 예약 건수", f"{total_bookings:,} 건")
    with col2: st.metric("순 취소율", f"{cancellation_rate:.2%}")
    with col3: st.metric("평균 ADR", f"$ {avg_adr:,.0f}")
    with col4: st.metric("평균 숙박 일수", f"{avg_stays:.1f} 일")
    st.markdown("---")

    st.header("🚨 Section A: 취소 및 리스크 관리")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.subheader("A-1. 월별 취소율 추이")
        if filtered_data['hotel'].nunique() > 1:
            hotel_cancel_trend = filtered_data.groupby(['arrival_date_month', 'hotel'])['is_canceled'].mean().reset_index()
            hotel_cancel_trend.columns = ['Month', 'Hotel Type', 'Cancellation Rate']
            overall_cancel_trend_df = filtered_data.groupby('arrival_date_month')['is_canceled'].mean().reset_index()
            overall_cancel_trend_df['Hotel Type'] = '통합 전체'
            overall_cancel_trend_df.columns = ['Month', 'Cancellation Rate', 'Hotel Type']
            combined_cancel_trend = pd.concat([hotel_cancel_trend, overall_cancel_trend_df])
        else:
            combined_cancel_trend = filtered_data.groupby('arrival_date_month')['is_canceled'].mean().reset_index()
            combined_cancel_trend.columns = ['Month', 'Cancellation Rate']
            combined_cancel_trend['Hotel Type'] = filtered_data['hotel'].iloc[0]
        fig_a1 = px.line(combined_cancel_trend, x='Month', y='Cancellation Rate', color='Hotel Type', title='월별 예약 취소율', markers=True, color_discrete_sequence=px.colors.qualitative.Pastel, height=400)
        fig_a1.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig_a1, use_container_width=True)
    with col_a2:
        st.subheader("A-2. 취소율 vs. 보증금 유형 및 리드타임 그룹")
        risk_analysis = filtered_data.groupby(['deposit_type', 'lead_time_group'])['is_canceled'].mean().reset_index()
        risk_analysis.columns = ['Deposit Type', 'Lead Time Group', 'Cancellation Rate']
        fig_a2 = px.bar(risk_analysis, x='Deposit Type', y='Cancellation Rate', color='Lead Time Group', barmode='group', title='보증금 및 선행기간 그룹별 취소 위험도', color_discrete_sequence=px.colors.qualitative.Pastel, height=400)
        fig_a2.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig_a2, use_container_width=True)

    st.subheader("A-3. 리드타임이 취소에 미치는 영향")
    try: plt.rcParams['font.family'] = 'Malgun Gothic'
    except: plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    lt_max = filtered_data['lead_time'].quantile(0.99)
    plot_data_density = filtered_data[filtered_data['lead_time'] <= lt_max].copy()
    plot_data_density['is_canceled_label'] = plot_data_density['is_canceled'].map({0: '미취소 (0)', 1: '취소 (1)'})
    fig, ax = plt.subplots(figsize=(10, 5))
    pastel_colors = {'미취소 (0)': '#85c793', '취소 (1)': '#ffba49'}
    sns.kdeplot(data=plot_data_density, x='lead_time', hue='is_canceled_label', hue_order=['미취소 (0)', '취소 (1)'], palette=pastel_colors, fill=True, alpha=.7, linewidth=2, ax=ax)
    ax.set_title('취소 여부별 예약 선행 기간(Lead Time) 분포 밀도')
    ax.set_xlabel('예약 선행 기간 (Lead Time)')
    ax.set_ylabel('밀도')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.header("💰 Section B: 재무 및 수익 분석")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.subheader("B-1. 월별 평균 ADR 추이")
        adr_trend = filtered_data.groupby('arrival_date_month')['adr'].mean().reset_index()
        adr_trend.columns = ['Month', 'Average ADR']
        fig_b1 = px.line(adr_trend, x='Month', y='Average ADR', title='월별 평균 일일 요금 (ADR)', markers=True, color_discrete_sequence=px.colors.qualitative.Pastel, height=400)
        fig_b1.update_yaxes(tickformat="$,.0f")
        st.plotly_chart(fig_b1, use_container_width=True)
    with col_b2:
        st.subheader("B-2. 시장 부문별 예상 수익")
        revenue_by_segment = filtered_data.groupby('market_segment')['estimated_revenue'].sum().reset_index()
        revenue_by_segment.columns = ['Market Segment', 'Total Estimated Revenue']
        revenue_by_segment = revenue_by_segment.sort_values('Total Estimated Revenue', ascending=False)
        fig_b2 = px.bar(revenue_by_segment, x='Market Segment', y='Total Estimated Revenue', title='시장 부문별 총 예상 수익', color='Market Segment', color_discrete_sequence=px.colors.qualitative.Pastel, height=400)
        fig_b2.update_yaxes(tickformat="$,.0f")
        st.plotly_chart(fig_b2, use_container_width=True)

    st.markdown("---")
    st.header("⚙️ Section C: 운영 효율 및 자원 관리")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("C-1. 변경 예약 건수 분포")
        changes_counts = filtered_data['booking_changes'].value_counts().reset_index()
        changes_counts.columns = ['Changes Count', 'Count']
        top_n = 5
        other_count = changes_counts[changes_counts['Changes Count'] >= top_n]['Count'].sum()
        changes_plot = changes_counts[changes_counts['Changes Count'] < top_n].copy()
        changes_plot.loc[len(changes_plot)] = [f'{top_n}+ 이상', other_count]
        fig_c1 = px.bar(changes_plot, x='Changes Count', y='Count', color_discrete_sequence=px.colors.qualitative.Pastel, title='예약 변경 건수 빈도 (프론트 부하)')
        st.plotly_chart(fig_c1, use_container_width=True)
    with col_c2:
        st.subheader("C-2. 특별 요청 건수 분포")
        fig_c2 = px.histogram(filtered_data, x='total_of_special_requests', color='is_canceled', barmode='group', color_discrete_sequence=px.colors.qualitative.Pastel, title='총 특별 요청 건수 빈도 (하우스키핑/컨시어지 부하)')
        fig_c2.update_xaxes(categoryorder='total ascending')
        st.plotly_chart(fig_c2, use_container_width=True)
    st.markdown("---")
    col_c3, col_c4 = st.columns(2)
    with col_c3:
        st.subheader("C-3. 주말 vs. 주중 숙박 비율 분포")
        stays_data = filtered_data[['stays_in_weekend_nights', 'stays_in_week_nights', 'hotel']].copy()
        stays_data_melt = stays_data.melt(id_vars=['hotel'], value_vars=['stays_in_weekend_nights', 'stays_in_week_nights'], var_name='Stay Type', value_name='Nights')
        stays_data_melt = stays_data_melt[stays_data_melt['Nights'] > 0]
        fig_c3 = px.box(stays_data_melt, x='Stay Type', y='Nights', color='hotel', points="outliers", title='호텔 유형별 주말/주중 숙박 일수 분포', color_discrete_sequence=px.colors.qualitative.Pastel, height=450)
        max_nights_95 = stays_data_melt['Nights'].quantile(0.95)
        fig_c3.update_yaxes(range=[0, max_nights_95])
        st.plotly_chart(fig_c3, use_container_width=True)
    with col_c4:
        st.subheader("C-4. 객실 타입 사용 현황 (상위 4개 + 기타)")

        top_4_rooms = filtered_data['reserved_room_type'].value_counts().nlargest(4).index.tolist()

        data_for_pie = filtered_data[['reserved_room_type']].copy()
        # --- 오류 수정 부분: np.where의 첫 번째 인자 수정 ---
        data_for_pie['room_group'] = np.where(
            data_for_pie['reserved_room_type'].isin(top_4_rooms), # 여기를 이렇게 수정합니다.
            data_for_pie['reserved_room_type'],
            '기타 (Other)'
        )
        # ----------------------------------------------------
        room_counts = data_for_pie['room_group'].value_counts().reset_index()
        room_counts.columns = ['Reserved Room Group', 'Count']
        fig_c4 = px.pie(room_counts, names='Reserved Room Group', values='Count', title='전체 예약 건수 대비 객실 타입별 점유율', color_discrete_sequence=px.colors.qualitative.Pastel, height=450)
        fig_c4.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_c4, use_container_width=True)

    st.markdown("---")
    st.header("D. 필터 적용 상세 데이터")
    st.dataframe(filtered_data.head(1000).drop(columns=['total_guests', 'total_stays', 'lead_time_group'], errors='ignore'), use_container_width=True)


# -------------------- 3. 호텔 통합 분석 필터 UI (dashboard_admin.py에서 가져옴) --------------------
def get_full_dashboard_filtered_data(data):
    st.sidebar.header("통합 분석 필터 (City + Resort)")
    hotel_types = sorted(data['hotel'].unique())
    selected_hotels = st.sidebar.multiselect("호텔 유형 선택", hotel_types, default=hotel_types, key="admin_pg_hotel_type_filter_full")

    arrival_years = sorted(data['arrival_date_year'].unique())
    selected_years = st.sidebar.multiselect("도착 연도 선택", arrival_years, default=arrival_years, key="admin_pg_arrival_year_filter_full_content")

    return data[
        (data['hotel'].isin(selected_hotels)) &
        (data['arrival_date_year'].isin(selected_years))
    ].copy()

# -------------------- 4. City Hotel 전용 분석 필터 UI (dashboard_admin.py에서 가져옴) --------------------
def get_city_hotel_filtered_data(data):
    st.sidebar.header("City Hotel 전용 필터")
    city_data = data[data['hotel'] == 'City Hotel'].copy()

    arrival_years = sorted(city_data['arrival_date_year'].unique())
    selected_years = st.sidebar.multiselect("도착 연도 선택", arrival_years, default=arrival_years, key="admin_pg_arrival_year_filter_city_content")

    month_order = city_data['arrival_date_month'].cat.categories.tolist()
    selected_months = st.sidebar.multiselect("도착 월 선택", month_order, default=month_order, key="admin_pg_arrival_month_filter_city_content")

    return city_data[
        (city_data['arrival_date_year'].isin(selected_years)) &
        (city_data['arrival_date_month'].isin(selected_months))
    ].copy()

