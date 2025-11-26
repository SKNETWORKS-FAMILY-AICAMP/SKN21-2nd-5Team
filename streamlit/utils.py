# streamlit/utils.py
import streamlit as st
import os

def logout():
    """사용자를 로그아웃시키고 로그인 페이지로 리다이렉트합니다."""
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None
    # 로그아웃 시 모든 접근 거부 메시지 상태를 초기화합니다.
    if 'access_denied_message' in st.session_state:
        del st.session_state.access_denied_message
    if 'denied_page_info' in st.session_state:
        del st.session_state.denied_page_info
    st.switch_page("pages/login.py")
    st.stop() # logout 함수 내부에 st.stop()이 있습니다.

def display_access_denied_message_once(current_page_name):
    """
    세션 상태에 저장된 접근 거부 메시지가 있다면 현재 페이지에서 표시하고 제거합니다.
    각 페이지 상단에서 호출되어야 합니다.
    """
    if 'access_denied_message' in st.session_state:
        denied_info = st.session_state.get('denied_page_info', {})

        # 메시지가 현재 페이지를 대상으로 하는 경우에만 표시
        # 참고: login.py에서 메시지를 표시하고 싶다면, login.py에서도 이 함수를 호출해야 합니다.
        if denied_info.get("to_page_name") == current_page_name or \
           (denied_info.get("to_page_name") == "login.py" and current_page_name == "login.py"):
            st.error(st.session_state.access_denied_message)
            # 메시지를 표시한 후 세션 상태에서 제거합니다.
            del st.session_state.access_denied_message
            del st.session_state.denied_page_info

def check_access(required_roles, current_page_name):
    """
    현재 사용자의 로그인 상태와 역할을 확인하고 접근을 제어합니다.
    - 로그인되지 않았으면 로그인 페이지로 리다이렉트 (로그아웃 처리).
    - 로그인되어 있지만 권한이 없으면 자신의 기본 페이지로 리다이렉트 (로그아웃 안함).
    - admin은 모든 페이지에 접근 가능.
    **이 함수는 UI 요소를 렌더링하지 않습니다. 오직 논리적인 접근 제어만 담당합니다.**
    """
    # 기본 페이지 매핑 (switch_page에 사용할 파일명)
    default_pages = {
        "admin": "pages/admin_pg.py", # Admin의 기본 페이지는 admin_pg.py (혹은 main.py)
        "guest": "pages/guest_pg.py"
    }

    # required_roles가 단일 문자열인 경우 리스트로 변환
    if not isinstance(required_roles, list):
        required_roles = [required_roles]

    # 1. 로그인 여부 확인
    if not st.session_state.get('logged_in', False):
        # 로그인되지 않은 경우, 메시지를 세션에 저장하고 로그인 페이지로 리다이렉트
        st.session_state.access_denied_message = "로그인이 필요합니다."
        st.session_state.denied_page_info = {
            "from_page_name": current_page_name,
            "to_page_name": "login.py"
        }
        logout() # logout() 함수 내에 st.switch_page와 st.stop() 포함

    current_role = st.session_state.get('role')

    # 2. Admin은 모든 페이지에 접근 허용
    if current_role == "admin":
        # Admin이더라도 이 함수는 사이드바를 렌더링하지 않습니다.
        # 사이드바 렌더링은 각 페이지 파일에서 담당합니다.
        return True # Admin은 접근 허용

    # 3. Admin이 아닌 경우, 페이지 역할 요구사항 확인
    if current_role not in required_roles:
        # 권한이 없는 경우, 메시지를 세션에 저장하고 사용자의 기본 페이지로 리다이렉트
        st.session_state.access_denied_message = f"접근 권한이 없습니다. 이 페이지는 {', '.join(required_roles)} 역할 사용자만 접근할 수 있습니다."
        st.session_state.denied_page_info = {
            "from_page_name": current_page_name,
            "to_page_name": default_pages.get(current_role, "pages/login.py")
        }

        if current_role in default_pages:
            st.switch_page(default_pages[current_role])
        else:
            # 알 수 없는 역할인 경우 로그인 페이지로 리다이렉트 (로그아웃 처리)
            logout()
        st.stop() # switch_page 또는 logout 후 현재 스크립트 실행 중지를 위해 필요

    # 4. 접근이 허용된 경우 (admin이 아니고, 역할이 일치하는 경우)
    # 이 경우에도 사이드바를 렌더링하지 않습니다.
    return True
