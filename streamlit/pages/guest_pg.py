import streamlit as st
import datetime
import os
import io
from PIL import Image
import pandas as pd
from utils import logout, check_access, display_access_denied_message_once # logout 함수가 이미 임포트되어 있습니다.
import sys
import lightgbm as lgb
import pickle

current_page_name = os.path.basename(__file__) # 현재 페이지 스크립트 이름 (예: "front_josh.py")

# 세션 상태에 저장된 접근 거부 메시지가 있다면 표시
display_access_denied_message_once(current_page_name)

# 페이지 로드 시 가장 먼저 접근 권한 확인
# check_access 함수 호출 시 두 번째 인자로 current_page_name을 전달합니다.
check_access(["admin", "guest"], current_page_name) # 이 페이지는 "admin" 또는 "guest"가 접근 가능

# --- 이미지 파일의 기본 경로 설정 ---
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# CSV_FILE_PATH를 current_dir과 같은 위치에 두는 것이 권장됩니다.
CSV_FILE_PATH = os.path.join(current_dir, '..', 'data',"hotel_bookings_data.csv") 
IMAGE_BASE_DIR = os.path.join(current_dir, '..', 'data', 'images') # 이미지 폴더 경로는 그대로


# --- 객실 유형 데이터  ---
room_type_definitions = {
    "A - $15": {
        "name": "A",
        "description": "A - $15",
        "image_filenames": ["A_1.jpg", "A_2.jpg", "A_3.jpg"]
    },
    "B - $25": {
        "name": "B",
        "description": "B - $25",
        "image_filenames": ["B_1.jpg", "B_2.jpg", "B_3.jpg"]
    },
    "C - $40": {
        "name": "C",
        "description": "C - $40",
        "image_filenames": ["C_1.jpg", "C_2.jpg", "C_3.jpg"]
    },
    "D - $65": {
        "name": "D",
        "description": "D - $65",
        "image_filenames": ["D_1.jpg", "D_2.jpg", "D_3.jpg"]
    },
    "E - $80": {"name": "E", "description": "E - $80", "image_filenames": ["E_1.jpg", "E_2.jpg", "E_3.jpg"]},
    "F - $100": {"name": "F", "description": "F - $100", "image_filenames": ["F_1.jpg", "F_2.jpg", "F_3.jpg"]},
    "G - $125": {"name": "G", "description": "G - $125", "image_filenames": ["G_1.jpg", "G_2.jpg", "G_3.jpg"]},
    "H - $140": {"name": "H", "description": "H - $140", "image_filenames": ["H_1.jpg", "H_2.jpg", "H_3.jpg"]},
    "I - $165": {"name": "I", "description": "I - $165", "image_filenames": ["I_1.jpg", "I_2.jpg", "I_3.jpg"]},
    "K - $185": {"name": "K", "description": "K - $185", "image_filenames": ["K_1.jpg", "K_2.jpg", "K_3.jpg"]},
    "L - $210": {"name": "L", "description": "L - $210", "image_filenames": ["L_1.jpg", "L_2.jpg", "L_3.jpg"]},
    "P - $250": {"name": "P", "description": "P - $250", "image_filenames": ["P_1.jpg", "P_2.jpg", "P_3.jpg"]}
}

# --- 이미지 크기 조절 및 크롭 함수 ---
def resize_and_crop_image(image_bytes, target_aspect_ratio=(16, 9)):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        img_width, img_height = img.size
        img_aspect_ratio = img_width / img_height
        
        target_width_ratio, target_height_ratio = target_aspect_ratio
        target_ratio = target_width_ratio / target_height_ratio

        if img_aspect_ratio > target_ratio:
            new_width = int(img_height * target_ratio)
            left = (img_width - new_width) / 2
            top = 0
            right = (img_width + new_width) / 2
            bottom = img_height
        else:
            new_height = int(img_width / target_ratio)
            left = 0
            top = (img_height - new_height) / 2
            right = img_width
            bottom = (img_height + new_height) / 2
            
        img_cropped = img.crop((left, top, right, bottom))
        
        output_bytes = io.BytesIO()
        img_cropped.save(output_bytes, format=img.format if img.format else "JPEG")
        output_bytes.seek(0)
        return output_bytes
    except Exception as e:
        st.error(f"이미지 처리 중 오류 발생: {e}")
        return None 

#--- 예측 함수 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
from modeling import predict
def pred(row):
    # 'row'가 Series(단일 레코드)일 경우 DataFrame으로 변환하여 일관된 처리를 보장
    if isinstance(row, pd.Series):
        cb_data = pd.DataFrame([row])
    else:
        cb_data = row.copy() # 이미 DataFrame인 경우 복사

    cb_data_name = cb_data['name'] # 이는 Series가 됩니다.
    cb_data['client_id'] = cb_data['name']

    cb_data_for_prediction  =cb_data.drop(columns=['name','is_canceled','country', 'assigned_room_type','booking_changes','days_in_waiting_list','reservation_status','reservation_status_date'])
    processed_features, _ = predict.preprocess_test_data(cb_data_for_prediction)
    
    feature_cols_path = os.path.join(PROJECT_ROOT,'model', 'feature_columns.pkl')
    feature_cols_path = os.path.normpath(feature_cols_path)

    with open(feature_cols_path, "rb") as f:
        feature_columns = pickle.load(f)
    
    cb_data_aligned = predict.align_test_features(processed_features, feature_columns) # 변수명 변경 (중복 피함)
    
    model_path = os.path.join(PROJECT_ROOT,'model', 'lgbm_model.txt')
    model_path = os.path.normpath(model_path)

    lgb_model = predict.load_model()
            
    predictions, probabilities = predict.predict_test_data(lgb_model, cb_data_aligned)
    
    output_path = os.path.join(PROJECT_ROOT,'data', 'test_predictions.csv')
    output_path = os.path.normpath(output_path)
    
    result_df = pd.DataFrame({
            'name': cb_data_name.tolist(), # Series를 list로 변환
            'prediction': predictions,
            'probability_no_cancel': [1 - p for p in probabilities], # 각 확률에 대해 1-p 계산
            'probability_cancel': probabilities
        })
    
    # --- 변경된 부분: header=False 추가 ---
    result_df.to_csv(output_path, mode='a', index=False, encoding='utf-8-sig', header=False)


# 페이지 설정 (전체 너비 사용)
st.set_page_config(
    layout="wide",
    page_title="호텔 예약 페이지",
    initial_sidebar_state="collapsed"
)

# --- 사이드바 내용 ---
with st.sidebar:
    
    # --- 로그아웃 버튼 ---
    if st.button("로그아웃", type="secondary", key="logout_guest_pg"):
        logout() # utils.py에서 가져온 logout 함수를 호출합니다.

st.title("🏨 호텔 예약 페이지")
st.markdown("---")

# 현재 날짜 및 내일 날짜 계산 (기본값 설정용)
today = datetime.date.today()
tomorrow = today + datetime.timedelta(days=1)

# --- Streamlit Session State 초기화 ---
if 'customer_name' not in st.session_state:
    st.session_state.customer_name = ""
if 'special_requests' not in st.session_state: # 추가 요청 사항 초기화
    st.session_state.special_requests = [{"id": 0, "text": ""}]
    st.session_state.next_request_id = 1

# 위젯의 초기값 설정을 위한 Session State 변수 초기화
if 'booking_date_range' not in st.session_state:
    st.session_state.booking_date_range = (today, tomorrow)
if 'adults_count' not in st.session_state:
    st.session_state.adults_count = 1
if 'children_count' not in st.session_state:
    st.session_state.children_count = 0
if 'infants_count' not in st.session_state:
    st.session_state.infants_count = 0
if 'room_type_selector' not in st.session_state:
    st.session_state.room_type_selector = "모든 객실 유형" # 초기값 설정
if 'meal_plan_selector' not in st.session_state:
    st.session_state.meal_plan_selector = "Undefined"
if 'parking_spaces_count' not in st.session_state:
    st.session_state.parking_spaces_count = 0

# 마지막 제출 결과 및 디버깅 정보를 저장할 세션 상태 변수
if 'last_submission_status' not in st.session_state:
    st.session_state.last_submission_status = None # 'success', 'error', None
if 'last_submission_data' not in st.session_state:
    st.session_state.last_submission_data = None # 저장된 디버깅 정보를 딕셔너리로 저장

# --- 제출 상태를 초기화하는 함수 ---
def clear_submission_status():
    st.session_state.last_submission_status = None
    st.session_state.last_submission_data = None

st.markdown("---")
# --- 고객님 성함, 체크인/아웃 날짜, 투숙객 수를 한 줄에 배치 ---
st.subheader("📝 예약 정보 입력")
col_name, col_date, col_guests = st.columns([1, 1.5, 1.5])

with col_name:
    st.markdown("### 고객님 성함")
    st.text_input(
        "성함을 입력해주세요:",
        placeholder="예: 홍길동",
        help="예약자분의 성함을 입력합니다.",
        key="customer_name",
        on_change=clear_submission_status # 변경 시 에러 메시지 초기화
    )

with col_date:
    st.markdown("### 체크인 / 체크아웃 날짜")
    booking_dates = st.date_input(
        "체크인/체크아웃 날짜를 선택해주세요:",
        min_value=today,
        key="booking_date_range", 
        help="체크인 날짜와 체크아웃 날짜를 선택합니다.",
        on_change=clear_submission_status # 변경 시 에러 메시지 초기화
    )

    check_in_date = None
    check_out_date = None
    if isinstance(booking_dates, tuple) and len(booking_dates) == 2:
        check_in_date = booking_dates[0]
        check_out_date = booking_dates[1]
    elif isinstance(booking_dates, datetime.date): # 단일 날짜 선택 시
        check_in_date = booking_dates
        check_out_date = booking_dates + datetime.timedelta(days=1)
    else: # 기본값 설정 (Session State가 아직 반영되지 않은 경우 등)
        check_in_date = today
        check_out_date = tomorrow
    
    # UI에 바로 에러 표시 (저장 버튼 누르기 전에도 볼 수 있도록)
    if check_in_date and check_out_date and check_out_date <= check_in_date:
        st.error("🚫 체크아웃 날짜는 체크인 날짜보다 늦어야 합니다. 날짜를 다시 확인해주세요.")


with col_guests:
    st.markdown("### 투숙객 수")
    adults_col_inner, children_col_inner, infants_col_inner = st.columns(3)
    with adults_col_inner:
        adults = st.number_input(
            "성인", min_value=0, key="adults_count", help="만 13세 이상의 투숙객 수",
            on_change=clear_submission_status # 변경 시 에러 메시지 초기화
        )
    with children_col_inner:
        children = st.number_input(
            "어린이", min_value=0, key="children_count", help="만 2세 ~ 12세 투숙객 수",
            on_change=clear_submission_status # 변경 시 에러 메시지 초기화
        )
    with infants_col_inner:
        infants = st.number_input(
            "유아", min_value=0, key="infants_count", help="만 2세 미만 투숙객 수",
            on_change=clear_submission_status # 변경 시 에러 메시지 초기화
        )

# --- 추가 검색 조건 섹션 ---
st.markdown("---")
st.subheader("🔍 추가 검색 조건")

add_col1, add_col2, add_col3 = st.columns(3)

with add_col1:
    room_type_options = ["모든 객실 유형"] + list(room_type_definitions.keys())
    room_type_filter = st.selectbox(
        "객실 유형",
        room_type_options,
        key="room_type_selector", 
        help="선호하는 객실의 유형을 선택하세요.",
        on_change=clear_submission_status # 변경 시 에러 메시지 초기화
    )
    if room_type_filter != "모든 객실 유형":
        selected_type_info = room_type_definitions.get(room_type_filter)
        if selected_type_info and "description" in selected_type_info:
            st.markdown(f"**유형 설명:** {selected_type_info['description']}", help=f"선택하신 {room_type_filter} 유형 객실에 대한 설명입니다.")

with add_col2:
    meal_plan_options = ("BB", "HB", "FB", "SC", "Undefined")
    meal_plan = st.selectbox(
        "식사 여부",
        meal_plan_options,
        key="meal_plan_selector", 
        help="BB: Bed & Breakfast (일 1회 식사), HB: Half Board (일 2회 식사), FB: Full Board (일 3회 식사), SC: Self-Catering(식사 안함), Undefined(미정)",
        on_change=clear_submission_status # 변경 시 에러 메시지 초기화
    )
with add_col3:
    required_parking_spaces = st.number_input(
        "주차 공간 수", min_value=0, max_value=4, key="parking_spaces_count", help="필요한 주차 공간의 수를 입력하세요. 객실당 최대 4대",
        on_change=clear_submission_status # 변경 시 에러 메시지 초기화
    )

# --- 선택된 객실 유형의 이미지 표시 (별도의 넓은 공간) ---
if room_type_filter != "모든 객실 유형":
    st.markdown("---")
    selected_type_info = room_type_definitions.get(room_type_filter) 
    
    if selected_type_info: 
        st.subheader(f"🖼️ 선택된 {selected_type_info['name']} 객실 이미지")

        if "image_filenames" in selected_type_info and isinstance(selected_type_info["image_filenames"], list):
            image_cols = st.columns(3) 
            
            # 파일이 3개 미만일 경우 generic_room.jpg로 채움
            images_to_display = (selected_type_info["image_filenames"] + ["generic_room.jpg"] * 3)[:3]

            for i, img_filename in enumerate(images_to_display):
                with image_cols[i]:
                    image_path = os.path.join(IMAGE_BASE_DIR, img_filename)
                    caption_text = f"{selected_type_info['name']} - 이미지 {i+1}" if img_filename != "generic_room.jpg" else f"대체 이미지: {selected_type_info['name']}"
                    
                    image_data = None
                    if os.path.exists(image_path):
                        try:
                            with open(image_path, "rb") as f:
                                image_data = f.read()
                            
                            processed_image_bytes = resize_and_crop_image(image_data, target_aspect_ratio=(16, 9))
                            if processed_image_bytes:
                                st.image(processed_image_bytes, caption=caption_text, width='stretch')
                            else:
                                st.text(f"이미지 처리 실패: {caption_text}")

                        except Exception as e:
                            st.warning(f"경고: 이미지 '{image_path}'를 읽거나 처리하는 중 오류 발생: {e}")
                            st.text(f"오류: {caption_text}")
                    else:
                        st.warning(f"경고: 이미지 '{image_path}'를 찾을 수 없습니다. 대체 이미지를 표시합니다.")
                        generic_image_path = os.path.join(IMAGE_BASE_DIR, "generic_room.jpg")
                        if os.path.exists(generic_image_path):
                            try:
                                with open(generic_image_path, "rb") as f:
                                    generic_image_data = f.read()
                                processed_generic_image_bytes = resize_and_crop_image(generic_image_data, target_aspect_ratio=(16, 9))
                                if processed_generic_image_bytes:
                                    st.image(processed_generic_image_bytes, caption=f"대체 이미지: {caption_text}", width='stretch')
                                else:
                                    st.text(f"대체 이미지 처리 실패: {caption_text}")
                            except Exception:
                                st.text(f"대체 이미지도 읽을 수 없습니다: {caption_text}")
                        else:
                            st.text(f"이미지 파일을 찾을 수 없습니다: {caption_text}")
        else:
            st.warning(f"객실 유형 '{room_type_filter}'에 대한 이미지 정보가 불완전합니다. 이미지를 표시할 수 없습니다.")
    else:
        st.warning(f"객실 유형 '{room_type_filter}'에 대한 정보가 없습니다.")


st.markdown("---")

# --- 추가 요청 사항 섹션 ---
st.subheader("📝 추가 요청 사항")

requests_to_delete = []

for i, item in enumerate(st.session_state.special_requests):
    col_req, col_btn = st.columns([0.9, 0.1])
    with col_req:
        item["text"] = st.text_input(
            f"요청 #{i+1}:",
            value=item["text"], 
            key=f"special_request_{item['id']}",
            placeholder="예: 늦은 체크아웃 요청, 특정 층 배정 요청 등",
            on_change=clear_submission_status # 변경 시 에러 메시지 초기화
        )
    with col_btn:
        if len(st.session_state.special_requests) > 1:
            st.markdown("<br>", unsafe_allow_html=True) # 버튼이 위로 올라가지 않도록 여백 추가
            if st.button("삭제", key=f"delete_request_{item['id']}", on_click=clear_submission_status): # 삭제 시 에러 메시지 초기화
                requests_to_delete.append(item["id"])

if requests_to_delete:
    st.session_state.special_requests = [
        item for item in st.session_state.special_requests
        if item["id"] not in requests_to_delete
    ]
    if not st.session_state.special_requests: # 모두 삭제되었다면 빈 요청 필드 하나 유지
        st.session_state.special_requests.append({"id": st.session_state.next_request_id, "text": ""})
        st.session_state.next_request_id += 1
    st.rerun() # 동적 위젯 변경을 위해 필요

if st.button("➕ 추가 요청 사항 추가", key="add_new_request_btn", on_click=clear_submission_status): # 추가 시 에러 메시지 초기화
    new_request_id = st.session_state.next_request_id
    st.session_state.special_requests.append({"id": new_request_id, "text": ""})
    st.session_state.next_request_id += 1
    st.rerun() # 동적 위젯 변경을 위해 필요

# `total_of_special_requests` 필드에 들어갈 실제 요청 개수
filtered_requests_count = len([item["text"] for item in st.session_state.special_requests if item["text"].strip() != ""])
st.write(f"현재 총 추가 요청 사항: **{filtered_requests_count}**개")


st.markdown("---")

# --- '예약' 버튼을 누르면 CSV 파일에 데이터를 추가하는 로직 ---
if st.button("✅ 예약", type="primary"):
    validation_passed = True
    submission_error_message = ""

    # st.session_state.customer_name으로 직접 접근
    if not st.session_state.customer_name:
        submission_error_message = "고객님 성함을 입력해주세요."
        validation_passed = False
    elif check_in_date is None or check_out_date is None:
        submission_error_message = "체크인/체크아웃 날짜를 정확히 선택해주세요."
        validation_passed = False
    elif check_out_date <= check_in_date:
        submission_error_message = "체크아웃 날짜는 체크인 날짜보다 늦어야 합니다. 날짜를 다시 확인해주세요."
        validation_passed = False
    # '모든 객실 유형' 선택 시 저장 방지
    elif st.session_state.room_type_selector == "모든 객실 유형":
        submission_error_message = "객실 유형을 반드시 선택해주세요."
        validation_passed = False

    if not validation_passed:
        st.session_state.last_submission_status = "error"
        st.session_state.last_submission_data = {"error_message": submission_error_message}
        # 유효성 검사 실패 시, st.rerun() 없이 현재 스크립트 실행을 마무리하고 하단 피드백 섹션에서 에러 표시
    else:
        # --- CSV 저장 로직 시작 ---
        # "name" 컬럼 및 "customer_special_requests" 컬럼 추가
        column_names = [
           'name', 'hotel','is_canceled','lead_time','arrival_date_year','arrival_date_month','arrival_date_week_number',
           'arrival_date_day_of_month','stays_in_weekend_nights','stays_in_week_nights','adults','children','babies',
           'meal','country','market_segment','distribution_channel','is_repeated_guest','previous_cancellations',
           'previous_bookings_not_canceled','reserved_room_type','assigned_room_type','booking_changes',
           'deposit_type','agent','company','days_in_waiting_list','customer_type','adr','required_car_parking_spaces',
           'total_of_special_requests','reservation_status','reservation_status_date',
           'customer_special_requests' # 새롭게 추가된 컬럼
        ]

        # 계산 가능한 값들
        total_nights = (check_out_date - check_in_date).days
        lead_time = (check_in_date - datetime.date.today()).days
        arrival_year = check_in_date.year
        arrival_month = check_in_date.month
        arrival_week = check_in_date.isocalendar()[1] 
        arrival_day = check_in_date.day

        week_stay = 0
        weekend_stay = 0
        current_date_iter = check_in_date
        while current_date_iter < check_out_date:
            if current_date_iter.weekday() >= 5: # 5:토요일, 6:일요일
                weekend_stay += 1
            else:
                week_stay += 1
            current_date_iter += datetime.timedelta(days=1)
        
        room_type_code_for_csv = room_type_filter.split(' ')[0]

        avg_price_per_night = float(room_type_filter.split('$')[1].strip())
        
        # --- 고객 요청 사항 처리 ---
        # 비어있지 않은 요청 텍스트만 추출
        actual_special_requests = [item["text"].strip() for item in st.session_state.special_requests if item["text"].strip() != ""]
        # 추출된 요청들을 "/"로 연결, 요청이 없으면 빈 문자열
        customer_special_requests_str = "/".join(actual_special_requests)


        data = {
            'name': st.session_state.customer_name, # 고객 이름 추가
            'hotel': 'City Hotel', 
            'is_canceled': 0, 
            'lead_time': lead_time,
            'arrival_date_year': arrival_year,
            'arrival_date_month': arrival_month,
            'arrival_date_week_number': arrival_week,
            'arrival_date_day_of_month': arrival_day,
            'stays_in_weekend_nights': weekend_stay,
            'stays_in_week_nights': week_stay,
            'adults': adults,
            'children': children,
            'babies': infants,
            'meal': meal_plan,
            'country': 'KOR', 
            'market_segment': 'Online TA', 
            'distribution_channel': 'TA/TO', 
            'is_repeated_guest': 0, 
            'previous_cancellations': 0, 
            'previous_bookings_not_canceled': 0, 
            'reserved_room_type': room_type_code_for_csv,
            'assigned_room_type': room_type_code_for_csv, 
            'booking_changes': 0, 
            'deposit_type': 'No Deposit', 
            'agent': 0, # Agent ID가 주어지지 않았으므로 0
            'company': 0, # Company ID가 주어지지 않았으므로 0
            'days_in_waiting_list': 0, 
            'customer_type': 'Transient', # 기본값
            'adr': avg_price_per_night,
            'required_car_parking_spaces': required_parking_spaces,
            'total_of_special_requests': filtered_requests_count, # 변경된 부분: 실제 유효한 요청 수
            'reservation_status': 'New Input', 
            'reservation_status_date': datetime.date.today().strftime('%Y-%m-%d'),
            'customer_special_requests': customer_special_requests_str # 추가된 고객 요청 사항
        }

        df_new_row = pd.DataFrame([data], columns=column_names)
        
        # --- 디버깅 정보를 session_state에 저장 (저장 전 상태) ---
        st.session_state.last_submission_data = {
            "path": CSV_FILE_PATH,
            "path_dir_exists": os.path.exists(os.path.dirname(CSV_FILE_PATH)),
            "file_exists_before_write": os.path.exists(CSV_FILE_PATH),
            "file_size_before_write": os.path.getsize(CSV_FILE_PATH) if os.path.exists(CSV_FILE_PATH) else 0,
            "data_to_save": df_new_row.to_dict('records')[0] # DataFrame을 dict로 변환하여 저장
        }

        try:
            # 데이터 디렉토리가 없으면 생성 (에러 방지)
            os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)

            # CSV 파일이 존재하고 내용이 있으면 append 모드, 아니면 header 포함 write 모드
            if os.path.exists(CSV_FILE_PATH) and os.path.getsize(CSV_FILE_PATH) > 0:
                df_new_row.to_csv(CSV_FILE_PATH, mode='a', header=False, index=False, encoding='utf-8-sig')
            else:
                df_new_row.to_csv(CSV_FILE_PATH, mode='w', header=True, index=False, encoding='utf-8-sig')
            
            pred(df_new_row)

            st.session_state.last_submission_status = "success"
            st.session_state.last_submission_data["file_exists_after_write"] = os.path.exists(CSV_FILE_PATH)
            st.session_state.last_submission_data["file_size_after_write"] = os.path.getsize(CSV_FILE_PATH)
            st.session_state.last_submission_data["customer_name"] = st.session_state.customer_name # 성공 메시지에 사용할 이름 저장 (클리어 전에 저장)
            st.session_state.show_balloons_now = True # 성공 시 풍선 표시 플래그 설정
            
            # 모든 세션 상태 초기화 (다음 입력을 위해)
            if 'customer_name' in st.session_state:
                del st.session_state['customer_name']

            st.subheader("✅ 예약 성공!")
            st.success(f"🎉 '{st.session_state.last_submission_data['customer_name']}' 님의 예약 정보가 성공적으로 입력되었습니다!")
            st.info("새로운 정보를 입력하려면 아래 '초기화' 버튼을 눌러주세요.")
            
        except Exception as e:
            st.session_state.last_submission_status = "error"
            # 오류 발생 시 디버깅 정보에 에러 메시지 추가
            st.session_state.last_submission_data["error_message"] = str(e)
            st.session_state.last_submission_data["file_exists_after_write"] = os.path.exists(CSV_FILE_PATH)
            st.session_state.last_submission_data["file_size_after_write"] = os.path.getsize(CSV_FILE_PATH) if os.path.exists(CSV_FILE_PATH) else 0
            st.subheader("❌ 예약 실패!")
            error_msg = st.session_state.last_submission_data.get("error_message", "알 수 없는 오류")
            st.error(f"🚫 예약 정보 저장 중 오류가 발생했습니다 : {error_msg}")

#--- 초기화 버튼 -----
if st.button("초기화", key="clear_success_message", on_click=clear_submission_status): # 확인 시 에러 메시지 초기화
         # 초기화할 키 목록 (special_requests 제외)
        keys_to_reset = [
            "customer_name",
            "booking_date_range",
            "adults_count",
            "children_count",
            "infants_count",
            "room_type_selector",
            "meal_plan_selector",
            "parking_spaces_count"
        ]

        # 선택한 키만 삭제 → Streamlit이 위젯 기본값으로 초기화
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]

        st.session_state.special_requests = [{"id": st.session_state.next_request_id, "text": ""}]
        st.session_state.next_request_id += 1
        
        st.rerun()



#객실 이미지 출처


# 모든 객실 유형", "A - $15", "B - $25", "C - $40", "D - $65", "E - $80", "F - $100", "G - $125", "H - $140", 
#          "I - $165", "K - $185", "L - $210", "P - $250")


# A_1  https://www.korea.kr/news/reporterView.do?newsId=148886962
# A_2  https://news.kbs.co.kr/news/pc/view/view.do?ncd=2567044
# A_3  https://www.hankookilbo.com/News/Read/201511060468427657


# https://www.airbnb.co.kr/rooms/10178608?check_in=2025-11-28&check_out=2025-11-30&search_mode=regular_search&category_tag=Tag%3A8678&photo_id=206668225&source_impression_id=p3_1763961949_P39KKf1PlqnLLUaG&previous_page_section_name=1000&federated_search_id=1636e429-b311-4560-8ccb-c9b56bca6ffa
# B_1  
# B_2  
# B_3  


# https://www.airbnb.co.kr/rooms/32169441?check_in=2025-11-28&check_out=2025-11-29&search_mode=regular_search&adults=1&category_tag=Tag%3A8678&children=0&infants=0&pets=0&photo_id=2121993274&source_impression_id=p3_1763962183_P3IEUx0SGmyNJZTe&previous_page_section_name=1000&federated_search_id=a3579e43-3112-4005-ab54-b93913ba8fb4&modal=PHOTO_TOUR_SCROLLABLE
# C_1  
# C_2  
# C_3  


# https://www.airbnb.co.kr/rooms/1529428001507163107?adults=1&check_in=2025-11-28&check_out=2025-11-29&guests=1&search_mode=regular_search&source_impression_id=p3_1763962418_P3W29RAKbNAvxSQO&previous_page_section_name=1000&federated_search_id=ab537a4a-ad23-4aa3-9e7e-2917b392a5b5
# D_1  
# D_2  
# D_3  


# https://www.airbnb.co.kr/rooms/17531537?adults=1&check_in=2025-11-28&check_out=2025-11-29&guests=1&search_mode=regular_search&source_impression_id=p3_1763962418_P3Fv8ykORPQFtqwa&previous_page_section_name=1000&federated_search_id=ab537a4a-ad23-4aa3-9e7e-2917b392a5b5
# E_1  
# E_2  
# E_3  


# https://www.airbnb.co.kr/rooms/1537436978916520180?check_in=2025-11-28&check_out=2025-11-30&search_mode=regular_search&source_impression_id=p3_1763962624_P3PCVyRaFPMdY_0X&previous_page_section_name=1000&federated_search_id=a1dfd081-a17c-442c-b505-7fce3dfe3268
# F_1  
# F_2  
# F_3  


# https://www.airbnb.co.kr/rooms/1471594113480544078?check_in=2025-11-28&check_out=2025-11-30&search_mode=regular_search&source_impression_id=p3_1763962183_P37AjSEZADR4Byzn&previous_page_section_name=1000&federated_search_id=a3579e43-3112-4005-ab54-b93913ba8fb4
# G_1  
# G_2  
# G_3  


# https://www.airbnb.co.kr/rooms/46249802?adults=1&check_in=2025-11-28&check_out=2025-11-29&guests=1&search_mode=regular_search&source_impression_id=p3_1763962901_P3JNGbJ1HZ5zEXW9&previous_page_section_name=1000&federated_search_id=4a7f2406-6388-4c1b-8a27-9163d8ddcd2d
# H_1  
# H_2  
# H_3  


# https://www.airbnb.co.kr/rooms/44985001?adults=1&check_in=2025-11-28&check_out=2025-11-29&guests=1&search_mode=regular_search&source_impression_id=p3_1763962711_P3RqXuVUs9Iju48L&previous_page_section_name=1000&federated_search_id=edeba5de-180a-4115-8a2f-bf42ebf5e0ac
# I_1  
# I_2  
# I_3  


# https://www.airbnb.co.kr/rooms/1503706162455008872?adults=1&check_in=2025-11-28&check_out=2025-11-29&guests=1&search_mode=regular_search&source_impression_id=p3_1763963153_P3h-i1lkBf81llxI&previous_page_section_name=1000&federated_search_id=c648bdee-3d93-461d-819a-cfaf054ec65a
# K_1  
# K_2  
# K_3  


# https://www.airbnb.co.kr/rooms/1336018302214711577?adults=1&check_in=2025-11-28&check_out=2025-11-29&guests=1&search_mode=regular_search&source_impression_id=p3_1763962711_P3JrFMDJRE95L7gV&previous_page_section_name=1000&federated_search_id=edeba5de-180a-4115-8a2f-bf42ebf5e0ac
# L_1  
# L_2  
# L_3  


# https://www.airbnb.co.kr/rooms/1428144722922099477?adults=1&check_in=2025-11-28&check_out=2025-11-29&guests=1&search_mode=regular_search&source_impression_id=p3_1763962711_P3h3dQxejCubgT5k&previous_page_section_name=1000&federated_search_id=edeba5de-180a-4115-8a2f-bf42ebf5e0ac
# P_1  
# P_2  
# P_3  