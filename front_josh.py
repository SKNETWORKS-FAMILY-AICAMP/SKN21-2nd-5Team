import streamlit as st
import datetime
import os
import io
from PIL import Image # Pillow 라이브러리 임포트
import pandas as pd # pandas 라이브러리 임포트

# --- 이미지 파일의 기본 경로 설정 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# CSV_FILE_PATH를 current_dir과 같은 위치에 두는 것이 권장됩니다.
CSV_FILE_PATH = os.path.join(current_dir, 'data',"hotel_bookings_add.csv") 
IMAGE_BASE_DIR = os.path.join(current_dir, 'data', 'images') # 이미지 폴더 경로는 그대로

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
        return None 

# --- Room Type Definitions ---
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
    "I - $165": {"name": "J", "description": "I - $165", "image_filenames": ["I_1.jpg", "I_2.jpg", "I_3.jpg"]},
    "K - $185": {"name": "K", "description": "K - $185", "image_filenames": ["K_1.jpg", "K_2.jpg", "K_3.jpg"]},
    "L - $210": {"name": "L", "description": "L - $210", "image_filenames": ["L_1.jpg", "L_2.jpg", "L_3.jpg"]},
    "P - $250": {"name": "P", "description": "P - $250", "image_filenames": ["P_1.jpg", "P_2.jpg", "P_3.jpg"]}
}

for full_type_str in ["A - $15", "B - $25", "C - $40", "D - $65", "E - $80", "F - $100", "G - $125", "H - $140", "I - $165", "K - $185", "L - $210", "P - $250"]:
    if full_type_str not in room_type_definitions:
        room_type_definitions[full_type_str] = {
            "name": f"객실 유형 {full_type_str.split(' ')[0]}",
            "description": f"객실 유형 {full_type_str.split(' ')[0]}에 대한 상세 설명입니다.",
            "image_filenames": ["generic_room.jpg", "generic_room.jpg", "generic_room.jpg"]
        }
    elif "image_filenames" not in room_type_definitions[full_type_str]:
        if "image_filename" in room_type_definitions[full_type_str]:
            filename = room_type_definitions[full_type_str]["image_filename"]
            room_type_definitions[full_type_str]["image_filenames"] = [filename, filename, filename]
            del room_type_definitions[full_type_str]["image_filename"]
        else:
            room_type_definitions[full_type_str]["image_filenames"] = ["generic_room.jpg", "generic_room.jpg", "generic_room.jpg"]



# 페이지 설정 (전체 너비 사용)
st.set_page_config(
    layout="wide",
    page_title="호텔 예약 페이지",
    initial_sidebar_state="collapsed"
)

st.title("🏨 호텔 예약 페이지")
st.markdown("---")

# --- Streamlit Session State 초기화 ---
if 'customer_name' not in st.session_state:
    st.session_state.customer_name = ""
if 'special_requests' not in st.session_state: # 추가 요청 사항 초기화
    st.session_state.special_requests = [{"id": 0, "text": ""}]
    st.session_state.next_request_id = 1

# 새롭게 추가: 마지막 제출 결과 및 디버깅 정보를 저장할 세션 상태 변수
if 'last_submission_status' not in st.session_state:
    st.session_state.last_submission_status = None # 'success', 'error', None
if 'last_submission_data' not in st.session_state:
    st.session_state.last_submission_data = None # 저장된 디버깅 정보를 딕셔너리로 저장


st.markdown("---")
# --- 고객님 성함, 체크인/아웃 날짜, 투숙객 수를 한 줄에 배치 ---
st.subheader("📝 예약 정보 입력")
col_name, col_date, col_guests = st.columns([1, 1.5, 1.5])

with col_name:
    st.markdown("### 고객님 성함")
    st.session_state.customer_name = st.text_input(
        "성함을 입력해주세요:",
        value=st.session_state.customer_name,
        placeholder="예: 홍길동",
        help="예약자분의 성함을 입력합니다."
    )

with col_date:
    st.markdown("### 체크인 / 체크아웃 날짜")
    today = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)

    booking_dates = st.date_input(
        "체크인/체크아웃 날짜를 선택해주세요:",
        (today, tomorrow),
        min_value=today,
        key="booking_date_range",
        help="체크인 날짜와 체크아웃 날짜를 선택합니다.",
    )

    check_in_date = None
    check_out_date = None
    if isinstance(booking_dates, tuple) and len(booking_dates) == 2:
        check_in_date = booking_dates[0]
        check_out_date = booking_dates[1]
    elif isinstance(booking_dates, list) and len(booking_dates) == 2:
        check_in_date = booking_dates[0]
        check_out_date = booking_dates[1]
    elif isinstance(booking_dates, datetime.date):
        check_in_date = booking_dates
        check_out_date = booking_dates + datetime.timedelta(days=1)
    else:
        check_in_date = today
        check_out_date = tomorrow
    
    if check_in_date and check_out_date and check_out_date <= check_in_date:
        st.error("🚫 체크아웃 날짜는 체크인 날짜보다 늦어야 합니다. 날짜를 다시 확인해주세요.")


with col_guests:
    st.markdown("### 투숙객 수")
    adults_col_inner, children_col_inner, infants_col_inner = st.columns(3)
    with adults_col_inner:
        adults = st.number_input(
            "성인", min_value=1, value=1, max_value=10, key="adults_count", help="만 13세 이상의 투숙객 수"
        )
    with children_col_inner:
        children = st.number_input(
            "어린이", min_value=0, value=0, max_value=5, key="children_count", help="만 2세 ~ 12세 투숙객 수"
        )
    with infants_col_inner:
        infants = st.number_input(
            "유아", min_value=0, value=0, max_value=3, key="infants_count", help="만 2세 미만 투숙객 수"
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
        index=0,
        help="선호하는 객실의 유형을 선택하세요."
    )
    if room_type_filter != "모든 객실 유형":
        selected_type_info = room_type_definitions.get(room_type_filter)
        if selected_type_info and "description" in selected_type_info:
            st.markdown(f"**유형 설명:** {selected_type_info['description']}", help=f"선택하신 {room_type_filter} 유형 객실에 대한 설명입니다.")

with add_col2:
    meal_plan = st.selectbox(
        "식사 여부",
        ("BB", "HB", "FB", "SC", "Undefined"),
        key="meal_plan_selector",
        index=0,
        help="BB: Bed & Breakfast (일 1회 식사), HB: Half Board (일 2회 식사), FB: Full Board (일 3회 식사), SC: Self-Catering(식사 안함), Undefined(미정)"
    )
with add_col3:
    required_parking_spaces = st.number_input(
        "주차 공간 수", min_value=0, value=0, max_value=4, key="parking_spaces_count", help="필요한 주차 공간의 수를 입력하세요. 객실당 최대 4대"
    )

# --- 선택된 객실 유형의 이미지 표시 (별도의 넓은 공간) ---
if room_type_filter != "모든 객실 유형":
    st.markdown("---")
    selected_type_info = room_type_definitions.get(room_type_filter) 
    
    if selected_type_info: 
        st.subheader(f"🖼️ 선택된 {selected_type_info['name']} 객실 이미지")

        if "image_filenames" in selected_type_info and isinstance(selected_type_info["image_filenames"], list):
            image_cols = st.columns(3) 
            
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
                                st.image(processed_image_bytes, caption=caption_text, use_container_width=True)
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
                                    st.image(processed_generic_image_bytes, caption=f"대체 이미지: {caption_text}", use_container_width=True)
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
            placeholder="예: 늦은 체크아웃 요청, 특정 층 배정 요청 등"
        )
    with col_btn:
        if len(st.session_state.special_requests) > 1:
            st.markdown("<br>", unsafe_allow_html=True) # 버튼이 위로 올라가지 않도록 여백 추가
            if st.button("삭제", key=f"delete_request_{item['id']}"):
                requests_to_delete.append(item["id"])

if requests_to_delete:
    st.session_state.special_requests = [
        item for item in st.session_state.special_requests
        if item["id"] not in requests_to_delete
    ]
    if not st.session_state.special_requests:
        st.session_state.special_requests.append({"id": st.session_state.next_request_id, "text": ""})
        st.session_state.next_request_id += 1
    st.rerun()

if st.button("➕ 추가 요청 사항 추가", key="add_new_request_btn"):
    new_request_id = st.session_state.next_request_id
    st.session_state.special_requests.append({"id": new_request_id, "text": ""})
    st.session_state.next_request_id += 1
    st.rerun()

filtered_requests = [item["text"] for item in st.session_state.special_requests if item["text"].strip() != ""]
total_special_requests_count = len(filtered_requests)

st.write(f"현재 총 추가 요청 사항: **{total_special_requests_count}**개")


st.markdown("---")

# --- '저장' 버튼을 누르면 CSV 파일에 데이터를 추가하는 로직 ---
if st.button("✅ 저장", type="primary"):
    if not st.session_state.customer_name:
        st.error("🚫 고객님 성함을 입력해주세요.")
        st.session_state.last_submission_status = "error"
        st.session_state.last_submission_data = {"error_message": "고객님 성함을 입력해주세요."}
    elif check_in_date and check_out_date and check_out_date <= check_in_date:
        st.error("🚫 체크아웃 날짜는 체크인 날짜보다 늦어야 합니다. 날짜를 다시 확인해주세요.")
        st.session_state.last_submission_status = "error"
        st.session_state.last_submission_data = {"error_message": "체크아웃 날짜는 체크인 날짜보다 늦어야 합니다."}
    else:
        # --- CSV 저장 로직 시작 ---
        column_names = [
            '호텔', '취소여부', '리드타임(일)','도착연도','도착월','도착주차','도착일',
            '주말숙박일수','주중숙박일수','성인수','어린이수','아이수','식사여부','출신국',
            '예약방법','유통채널','재방문고객여부','고객과거취소건수','고객예약성공기록',
            '예약객실유형','배정객실유형','예약변경수','보증금유형','에이전트ID','회사ID',
            '예약대기목록에있었던기간','고객유형','1박당평균요금','요청한주차공간수','특수요청수',
            '예약상태','예약상태가최종업데이트날짜'
        ]

        # 계산 가능한 값들
        # check_in_date 또는 check_out_date가 None일 경우를 대비하여 오류 체크를 더 강화
        if check_in_date is None or check_out_date is None:
            st.error("🚫 체크인/체크아웃 날짜를 정확히 선택해주세요.")
            st.session_state.last_submission_status = "error"
            st.session_state.last_submission_data = {"error_message": "체크인/체크아웃 날짜가 유효하지 않습니다."}
            st.rerun() # 오류 발생 시 즉시 재실행하여 메시지 표시
        
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
            if current_date_iter.weekday() >= 5:
                weekend_stay += 1
            else:
                week_stay += 1
            current_date_iter += datetime.timedelta(days=1)
        
        room_type_code_for_csv = "Undefined"
        if room_type_filter != "모든 객실 유형":
            room_type_code_for_csv = room_type_filter.split(' ')[0]

        avg_price_per_night = 0
        if room_type_code_for_csv != "Undefined":
            found_room_for_price = next(
                (room for room in sample_rooms if room.get("type_code") == room_type_code_for_csv), 
                None
            )
            if found_room_for_price and "price_per_night" in found_room_for_price:
                avg_price_per_night = found_room_for_price['price_per_night']
            else:
                st.warning(f"경고: 객실 유형 '{room_type_code_for_csv}'에 대한 샘플 가격 정보를 찾을 수 없습니다. 1박당 평균 요금은 0으로 기록됩니다.")

        data = {
            '호텔': 'City Hotel', 
            '취소여부': 0, 
            '리드타임(일)': lead_time,
            '도착연도': arrival_year,
            '도착월': arrival_month,
            '도착주차': arrival_week,
            '도착일': arrival_day,
            '주말숙박일수': weekend_stay,
            '주중숙박일수': week_stay,
            '성인수': adults,
            '어린이수': children,
            '아이수': infants,
            '식사여부': meal_plan,
            '출신국': 'KOR', 
            '예약방법': 'Online TA', 
            '유통채널': 'TA/TO', 
            '재방문고객여부': 0, 
            '고객과거취소건수': 0, 
            '고객예약성공기록': 0, 
            '예약객실유형': room_type_code_for_csv,
            '배정객실유형': room_type_code_for_csv,
            '예약변경수': 0, 
            '보증금유형': 'No Deposit', 
            '에이전트ID': 0, 
            '회사ID': 0, 
            '예약대기목록에있었던기간': 0, 
            '고객유형': 'Transient', 
            '1박당평균요금': avg_price_per_night,
            '요청한주차공간수': required_parking_spaces,
            '특수요청수': total_special_requests_count,
            '예약상태': 'New Input', 
            '예약상태가최종업데이트날짜': datetime.date.today().strftime('%Y-%m-%d')
        }

        df_new_row = pd.DataFrame([data], columns=column_names)
        
        # --- 디버깅 정보를 session_state에 저장 ---
        st.session_state.last_submission_data = {
            "path": CSV_FILE_PATH,
            "path_dir_exists": os.path.exists(os.path.dirname(CSV_FILE_PATH)),
            "file_exists_before_write": os.path.exists(CSV_FILE_PATH),
            "file_size_before_write": os.path.getsize(CSV_FILE_PATH) if os.path.exists(CSV_FILE_PATH) else 0,
            "data_to_save": df_new_row.to_dict('records')[0] # DataFrame을 dict로 변환하여 저장
        }

        try:
            if os.path.exists(CSV_FILE_PATH) and os.path.getsize(CSV_FILE_PATH) > 0:
                df_new_row.to_csv(CSV_FILE_PATH, mode='a', header=False, index=False, encoding='utf-8-sig')
            else:
                df_new_row.to_csv(CSV_FILE_PATH, mode='w', header=True, index=False, encoding='utf-8-sig')
            
            st.session_state.last_submission_status = "success"
            st.session_state.last_submission_data["file_exists_after_write"] = os.path.exists(CSV_FILE_PATH)
            st.session_state.last_submission_data["file_size_after_write"] = os.path.getsize(CSV_FILE_PATH)
            st.session_state.last_submission_data["customer_name"] = st.session_state.customer_name # 성공 메시지에 사용할 이름 저장
            
            # 모든 세션 상태 초기화 (다음 입력을 위해)
            st.session_state.customer_name = ""
            st.session_state.special_requests = [{"id": 0, "text": ""}]
            st.session_state.next_request_id = 1
            st.rerun() # 성공 메시지와 초기화된 폼을 표시하기 위해 재실행
            
        except Exception as e:
            st.session_state.last_submission_status = "error"
            st.session_state.last_submission_data["error_message"] = str(e)
            st.session_state.last_submission_data["file_exists_after_write"] = os.path.exists(CSV_FILE_PATH)
            st.session_state.last_submission_data["file_size_after_write"] = os.path.getsize(CSV_FILE_PATH) if os.path.exists(CSV_FILE_PATH) else 0
            st.rerun() # 에러 메시지를 표시하기 위해 재실행

# --- 스크립트 실행 후, 지속적인 피드백 및 디버깅 정보 표시 ---
if st.session_state.last_submission_status == "success":
    st.balloons()
    st.subheader("✅ 예약 저장 성공!")
    st.success(f"🎉 '{st.session_state.last_submission_data['customer_name']}' 님의 예약 정보가 성공적으로 입력되었습니다!")
    st.info("새로운 정보를 입력하려면 아래 '확인 및 초기화' 버튼을 눌러주세요.")
    
    st.subheader("🔗 최근 저장된 디버깅 정보")
    data = st.session_state.last_submission_data
    st.write(f"**CSV 저장 경로:** `{data['path']}`")
    st.write(f"**상위 디렉토리 존재 여부:** `{data['path_dir_exists']}`")
    st.write(f"**쓰기 전 파일 존재 여부:** `{data['file_exists_before_write']}` (크기: `{data['file_size_before_write']}` bytes)")
    st.write(f"**쓰기 후 파일 존재 여부:** `{data['file_exists_after_write']}` (크기: `{data['file_size_after_write']}` bytes)")
    st.write("**저장된 데이터:**")
    st.json(data['data_to_save']) # 딕셔너리 형태로 보기 쉽게 출력

    if st.button("확인 및 초기화", key="clear_success_message"):
        st.session_state.last_submission_status = None
        st.session_state.last_submission_data = None
        st.rerun()

elif st.session_state.last_submission_status == "error":
    st.subheader("❌ 예약 저장 실패!")
    error_msg = st.session_state.last_submission_data.get("error_message", "알 수 없는 오류")
    st.error(f"🚫 CSV 파일 저장 중 오류가 발생했습니다: {error_msg}")
    st.warning("파일 경로, 쓰기 권한, 또는 파일이 다른 프로그램에 의해 잠겨 있는지 확인해 주세요.")

    st.subheader("🔗 오류 발생 시 디버깅 정보")
    data = st.session_state.last_submission_data
    st.write(f"**CSV 저장 경로:** `{data.get('path', '정보 없음')}`")
    st.write(f"**상위 디렉토리 존재 여부:** `{data.get('path_dir_exists', '정보 없음')}`")
    st.write(f"**쓰기 전 파일 존재 여부:** `{data.get('file_exists_before_write', '정보 없음')}` (크기: `{data.get('file_size_before_write', 0)}` bytes)")
    st.write(f"**쓰기 후 파일 존재 여부:** `{data.get('file_exists_after_write', '정보 없음')}` (크기: `{data.get('file_size_after_write', 0)}` bytes)")
    st.write("**시도된 데이터:**")
    if 'data_to_save' in data:
        st.json(data['data_to_save'])
    else:
        st.write("데이터 없음.")
    
    if st.button("확인", key="clear_error_message"):
        st.session_state.last_submission_status = None
        st.session_state.last_submission_data = None
        st.rerun()

"""

객실 이미지 출처


모든 객실 유형", "A - $15", "B - $25", "C - $40", "D - $65", "E - $80", "F - $100", "G - $125", "H - $140", 
         "I - $165", "K - $185", "L - $210", "P - $250")


A_1  https://www.korea.kr/news/reporterView.do?newsId=148886962
A_2  https://news.kbs.co.kr/news/pc/view/view.do?ncd=2567044
A_3  https://www.hankookilbo.com/News/Read/201511060468427657


https://www.airbnb.co.kr/rooms/10178608?check_in=2025-11-28&check_out=2025-11-30&search_mode=regular_search&category_tag=Tag%3A8678&photo_id=206668225&source_impression_id=p3_1763961949_P39KKf1PlqnLLUaG&previous_page_section_name=1000&federated_search_id=1636e429-b311-4560-8ccb-c9b56bca6ffa
B_1  
B_2  
B_3  


https://www.airbnb.co.kr/rooms/32169441?check_in=2025-11-28&check_out=2025-11-29&search_mode=regular_search&adults=1&category_tag=Tag%3A8678&children=0&infants=0&pets=0&photo_id=2121993274&source_impression_id=p3_1763962183_P3IEUx0SGmyNJZTe&previous_page_section_name=1000&federated_search_id=a3579e43-3112-4005-ab54-b93913ba8fb4&modal=PHOTO_TOUR_SCROLLABLE
C_1  
C_2  
C_3  


https://www.airbnb.co.kr/rooms/1529428001507163107?adults=1&check_in=2025-11-28&check_out=2025-11-29&guests=1&search_mode=regular_search&source_impression_id=p3_1763962418_P3W29RAKbNAvxSQO&previous_page_section_name=1000&federated_search_id=ab537a4a-ad23-4aa3-9e7e-2917b392a5b5
D_1  
D_2  
D_3  


https://www.airbnb.co.kr/rooms/17531537?adults=1&check_in=2025-11-28&check_out=2025-11-29&guests=1&search_mode=regular_search&source_impression_id=p3_1763962418_P3Fv8ykORPQFtqwa&previous_page_section_name=1000&federated_search_id=ab537a4a-ad23-4aa3-9e7e-2917b392a5b5
E_1  
E_2  
E_3  


https://www.airbnb.co.kr/rooms/1537436978916520180?check_in=2025-11-28&check_out=2025-11-30&search_mode=regular_search&source_impression_id=p3_1763962624_P3PCVyRaFPMdY_0X&previous_page_section_name=1000&federated_search_id=a1dfd081-a17c-442c-b505-7fce3dfe3268
F_1  
F_2  
F_3  


https://www.airbnb.co.kr/rooms/1471594113480544078?check_in=2025-11-28&check_out=2025-11-30&search_mode=regular_search&source_impression_id=p3_1763962183_P37AjSEZADR4Byzn&previous_page_section_name=1000&federated_search_id=a3579e43-3112-4005-ab54-b93913ba8fb4
G_1  
G_2  
G_3  


https://www.airbnb.co.kr/rooms/46249802?adults=1&check_in=2025-11-28&check_out=2025-11-29&guests=1&search_mode=regular_search&source_impression_id=p3_1763962901_P3JNGbJ1HZ5zEXW9&previous_page_section_name=1000&federated_search_id=4a7f2406-6388-4c1b-8a27-9163d8ddcd2d
H_1  
H_2  
H_3  


https://www.airbnb.co.kr/rooms/44985001?adults=1&check_in=2025-11-28&check_out=2025-11-29&guests=1&search_mode=regular_search&source_impression_id=p3_1763962711_P3RqXuVUs9Iju48L&previous_page_section_name=1000&federated_search_id=edeba5de-180a-4115-8a2f-bf42ebf5e0ac
I_1  
I_2  
I_3  


https://www.airbnb.co.kr/rooms/1503706162455008872?adults=1&check_in=2025-11-28&check_out=2025-11-29&guests=1&search_mode=regular_search&source_impression_id=p3_1763963153_P3h-i1lkBf81llxI&previous_page_section_name=1000&federated_search_id=c648bdee-3d93-461d-819a-cfaf054ec65a
K_1  
K_2  
K_3  


https://www.airbnb.co.kr/rooms/1336018302214711577?adults=1&check_in=2025-11-28&check_out=2025-11-29&guests=1&search_mode=regular_search&source_impression_id=p3_1763962711_P3JrFMDJRE95L7gV&previous_page_section_name=1000&federated_search_id=edeba5de-180a-4115-8a2f-bf42ebf5e0ac
L_1  
L_2  
L_3  


https://www.airbnb.co.kr/rooms/1428144722922099477?adults=1&check_in=2025-11-28&check_out=2025-11-29&guests=1&search_mode=regular_search&source_impression_id=p3_1763962711_P3h3dQxejCubgT5k&previous_page_section_name=1000&federated_search_id=edeba5de-180a-4115-8a2f-bf42ebf5e0ac
P_1  
P_2  
P_3  
"""