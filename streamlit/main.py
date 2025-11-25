import streamlit as st
from utils import logout # utils에서 logout 함수를 import 합니다.

st.set_page_config(
    page_title="메인 앱",
    initial_sidebar_state="collapsed"
)

# 세션 상태 초기화 (초기 앱 시작 시 한 번만 실행)
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None

# ----------------- 변경된 로직 -----------------
if not st.session_state.logged_in:
    # 로그인되지 않은 경우, 로그인 페이지로
    st.switch_page("pages/login.py")
else:
    # 로그인된 경우, main.py가 대시보드 역할을 합니다.
    st.title(f"환영합니다, {st.session_state.username}님! 👋")

    if st.session_state.role == "admin":
        st.subheader("관리자 메인 대시보드")
        st.write("여기는 관리자가 로그인 후 볼 수 있는 메인 대시보드입니다.")
        st.info("사이드바에서 'Admin 페이지'로 이동하여 상세 기능을 사용하세요.")
        # 관리자 전용 요약 정보나 빠른 링크 등을 여기에 추가할 수 있습니다.
        # st.page_link("pages/index.py", label="Admin 페이지로 바로가기")
    elif st.session_state.role == "guest":
        st.subheader("게스트 메인 페이지")
        st.write("여기는 게스트가 로그인 후 볼 수 있는 메인 페이지입니다.")
        st.info("사이드바에서 'Guest 페이지'로 이동하여 상세 기능을 사용하세요.")
        # 게스트 전용 요약 정보나 빠른 링크 등을 여기에 추가할 수 있습니다.
        # st.page_link("pages/front_josh.py", label="Guest 페이지로 바로가기")
    else:
        st.error("알 수 없는 역할입니다. 다시 로그인해주세요.")
        logout() # 알 수 없는 역할이면 로그아웃 처리

    # 로그인된 상태에서 사이드바에 로그아웃 버튼 표시 (utils.check_access 로직과 중복될 수 있으나, 메인 페이지의 경우 명시적으로 추가)
    with st.sidebar:
        st.write(f"로그인 사용자: {st.session_state.username} ({st.session_state.role})")
        if st.button("로그아웃", type="secondary"):
            logout()