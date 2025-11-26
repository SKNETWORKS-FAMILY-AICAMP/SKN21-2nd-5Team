# streamlit/main.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from utils import logout
import os

st.set_page_config(
    page_title="호텔 관리 앱",
    initial_sidebar_state="collapsed",
    layout="wide"
)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None

if not st.session_state.logged_in:
    st.switch_page("pages/login.py")
else:
    st.title(f"환영합니다, {st.session_state.username}님! 👋")

    if st.session_state.role == "admin":
        st.subheader("관리자 메인 페이지")
        st.write("여기는 관리자가 로그인 후 볼 수 있는 메인 페이지입니다.")
        # 이제 대시보드는 admin_pg.py 안에 있습니다.
        st.info("사이드바(왼쪽 화살표 클릭)에서 '통합 관리자 페이지'로 이동하여 상세 기능을 사용하세요.")

    elif st.session_state.role == "guest":
        st.subheader("게스트 메인 페이지")
        st.write("여기는 게스트가 로그인 후 볼 수 있는 메인 페이지입니다.")
        st.info("사이드바(왼쪽 화살표 클릭)에서 'Guest 페이지'로 이동하여 상세 기능을 사용하세요.")
    else:
        st.error("알 수 없는 역할입니다. 다시 로그인해주세요.")
        logout()

    with st.sidebar:
        # 이 블록 안에 아무것도 넣지 않으면 사이드바는 비어있습니다.
        pass # placeholder
