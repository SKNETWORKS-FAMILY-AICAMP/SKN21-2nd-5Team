import streamlit as st
import os 
from utils import logout, display_access_denied_message_once

# --- 이 부분을 추가/수정해야 합니다. ---
current_page_name = os.path.basename(__file__) # 현재 페이지 스크립트 이름 (예: "login.py")

# 세션 상태에 저장된 접근 거부 메시지가 있다면 표시 (로그인 페이지를 대상으로 하는 경우)
display_access_denied_message_once(current_page_name)
# -------------------------------------

# 이미 로그인된 상태에서 login.py에 접속하려 할 경우
# 해당 역할의 페이지로 자동 전환 (로그인 페이지를 볼 필요 없음)
if st.session_state.get('logged_in', False):
    # 로그인된 사용자는 역할에 관계없이 main.py (대시보드)로 이동
    st.switch_page("main.py") # <--- 이 부분을 수정했습니다.
    # 이전 코드:
    # if st.session_state.role == "admin":
    #     st.switch_page("pages/admin_pg.py")
    # elif st.session_state.role == "guest":
    #     st.switch_page("pages/guest_pg.py")
    # else:
    #     # 알 수 없는 역할이면 강제로 로그아웃 처리
    #     logout()
    #     st.stop()

st.title("로그인")

with st.form("login_form"):
    username = st.text_input("아이디")
    password = st.text_input("비밀번호", type="password")
    submitted = st.form_submit_button("로그인")

    if submitted:
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = "admin"
            st.success("관리자님, 성공적으로 로그인되었습니다!")
            # 로그인 성공 후 main.py (대시보드)로 이동
            st.switch_page("main.py")
        elif username == "guest" and password == "1234":
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = "guest"
            st.success("게스트님, 성공적으로 로그인되었습니다!")
            # 로그인 성공 후 main.py (대시보드)로 이동
            st.switch_page("main.py")
        else:
            st.error("잘못된 아이디 또는 비밀번호입니다.")