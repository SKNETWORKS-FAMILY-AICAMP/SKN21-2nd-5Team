# streamlit/pages/admin_pg.py
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

if streamlit_app_dir not in sys.path:
    sys.path.insert(0, streamlit_app_dir)

# --- 유틸리티 함수 임포트 ---
from admin_util import visualize_customer_data, get_customer_prediction, display_prediction_results
from utils import logout, check_access, display_access_denied_message_once # logout 함수가 이미 임포트되어 있습니다.

current_page_name = os.path.basename(__file__)
display_access_denied_message_once(current_page_name)
check_access("admin", current_page_name)

st.set_page_config(
    page_title="관리자 통합 분석", # 페이지 제목을 통합된 기능에 맞게 변경
    initial_sidebar_state="collapsed",
    layout="wide"
)

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
        data_for_pie['room_group'] = np.where(
            data_for_pie['reserved_room_type'].isin(top_4_rooms),
            data_for_pie['reserved_room_type'],
            '기타 (Other)'
        )
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
    # 이 함수는 사이드바에 필터 UI를 렌더링하고, 필터링된 데이터를 반환합니다.
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
    # 이 함수는 사이드바에 필터 UI를 렌더링하고, 필터링된 데이터를 반환합니다.
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


# -------------------- 통합된 사이드바 UI 및 기능 선택 --------------------
# 하나의 with st.sidebar 블록으로 모든 사이드바 요소를 정의
with st.sidebar:

    # 기능 선택 버튼들 (클릭 시 세션 상태 변경 및 reru)
    # 현재 페이지는 admin_pg.py 이므로 링크로 둘 필요 없이 버튼으로 기능 전환
    if st.button("👤 개별 고객 분석", key="admin_pg_customer_analysis"):
        st.session_state.current_admin_view = '개별 고객 분석'
        st.rerun()

    if st.button("📊 호텔 통계 대시보드", key="admin_pg_dashboard_analysis"):
        st.session_state.current_admin_view = '호텔 통계 대시보드'
        st.rerun()

    st.markdown("---")
    
    # --- 로그아웃 버튼 (utils.logout 함수를 호출) ---
    if st.button("로그아웃", type="secondary", key="logout_admin_pg_unified"):
        logout() # utils.py에서 가져온 logout 함수를 호출합니다.


# 초기 뷰 설정 (세션 상태 활용)
if 'current_admin_view' not in st.session_state:
    st.session_state.current_admin_view = '개별 고객 분석' # 기본값

# -------------------- 메인 컨텐츠 영역 렌더링 --------------------
pio.templates.default = "plotly_white"

if st.session_state.current_admin_view == '개별 고객 분석':
    st.title("🏨 개별 고객 정보 분석")
    st.markdown("---")

    if customer_df.empty:
        st.warning("고객 데이터를 로드하지 못했습니다. 파일 경로를 확인해 주세요.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### ✅ 고객 선택 방식")
        with col2:
            selection_method = st.radio(
                "선택 방식",
                ["테이블에서 행 선택", "고객 ID로 직접 검색"],
                label_visibility="collapsed",
                horizontal=True,
                key="customer_selection_method"
            )
        st.markdown("<br>", unsafe_allow_html=True)

        if selection_method == "고객 ID로 직접 검색":
            customer_ids = customer_df['name'].unique() if 'name' in customer_df.columns else []

            selected_customer_id = st.selectbox("분석할 고객 이름 선택", options=customer_ids, index=None, key="customer_id_select")

            if selected_customer_id:
                selected_row = customer_df[customer_df['name'] == selected_customer_id].iloc[0]
                selected_index = customer_df[customer_df['name'] == selected_customer_id].index[0]

                visualize_customer_data(selected_row, selected_index)

                prediction_data = get_customer_prediction(selected_row, predictions_df)
                display_prediction_results(prediction_data, predictions_df)

        else:
            st.subheader("🗂️ 고객 데이터 테이블")
            st.write("아래에서 행을 선택하여 상세 정보를 확인하세요")

            events = st.dataframe(
                data=customer_df,
                width='stretch',
                on_select="rerun",
                selection_mode="single-row",
                key="customer_data_table"
            )

            if events.selection["rows"]:
                selected_indices = events.selection["rows"]
                selected_index = selected_indices[0]
                selected_row = customer_df.iloc[selected_index]

                visualize_customer_data(selected_row, selected_index)

                prediction_data = get_customer_prediction(selected_row, predictions_df)
                display_prediction_results(prediction_data, predictions_df)

else: # 호텔 통계 대시보드
    if dashboard_data.empty:
        st.error("대시보드 데이터를 로드하지 못했습니다.")
        st.stop() # 데이터 로드 실패 시 페이지 실행 중단

    st.title("📊 호텔 통계 대시보드")
    st.markdown("---")

    # 대시보드 유형 선택 라디오 버튼 (사이드바에 렌더링)
    st.sidebar.header("대시보드 유형 선택")
    dashboard_type_selection = st.sidebar.radio(
        "유형",
        ['통합 분석 (City + Resort)', 'City Hotel 전용 분석'],
        key="admin_pg_dashboard_type_radio" # key를 명확히 구분
    )
    st.sidebar.markdown("---") # 대시보드 필터와 구분되는 선 추가

    # 선택된 대시보드 유형에 따라 필터링 및 차트 생성
    if dashboard_type_selection == '통합 분석 (City + Resort)':
        filtered_data_for_charts = get_full_dashboard_filtered_data(dashboard_data)
        generate_charts(filtered_data_for_charts, "🏨 호텔 예약 통계 분석 대시보드 (통합)")
    else: # City Hotel 전용 분석
        filtered_data_for_charts = get_city_hotel_filtered_data(dashboard_data)
        generate_charts(filtered_data_for_charts, "🏢 City Hotel 전용 분석 대시보드")
