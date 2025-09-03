"""
💊 Medication Agent - UI 컴포넌트
Streamlit UI 컴포넌트 및 세션 상태 관리를 담당합니다.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from config import SESSION_KEYS, PAGE_TITLE, PAGE_ICON, LAYOUT
from medication_db import MedicationDatabase
from langchain_agent import (
    initialize_openai, initialize_qdrant, initialize_embeddings,
    create_agent, initialize_qa_chain, check_qdrant_data,
    initialize_qdrant_db, reindex_qdrant_data
)


def initialize_session_state():
    """세션 상태를 초기화합니다."""
    if SESSION_KEYS["messages"] not in st.session_state:
        st.session_state[SESSION_KEYS["messages"]] = []
    
    if SESSION_KEYS["conversation_id"] not in st.session_state:
        st.session_state[SESSION_KEYS["conversation_id"]] = str(uuid.uuid4())
    
    if SESSION_KEYS["current_user"] not in st.session_state:
        st.session_state[SESSION_KEYS["current_user"]] = None
    
    if SESSION_KEYS["med_db"] not in st.session_state:
        st.session_state[SESSION_KEYS["med_db"]] = None
    
    if SESSION_KEYS["agent"] not in st.session_state:
        st.session_state[SESSION_KEYS["agent"]] = None
    
    if SESSION_KEYS["qdrant_client"] not in st.session_state:
        st.session_state[SESSION_KEYS["qdrant_client"]] = None
    
    if SESSION_KEYS["qa_chain"] not in st.session_state:
        st.session_state[SESSION_KEYS["qa_chain"]] = None
    
    if SESSION_KEYS["test_mode"] not in st.session_state:
        st.session_state[SESSION_KEYS["test_mode"]] = False


def add_message(role: str, content: str, sources: List[Dict] = None):
    """
    메시지를 세션 상태에 추가합니다.
    
    Args:
        role (str): 메시지 역할 (user/assistant)
        content (str): 메시지 내용
        sources (List[Dict]): 소스 문서 (선택사항)
    """
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    
    if sources:
        message["sources"] = sources
    
    st.session_state[SESSION_KEYS["messages"]].append(message)


def render_sidebar():
    """사이드바를 렌더링합니다."""
    with st.sidebar:
        st.title("⚙️ 설정")
        
        # 사용자 정보 입력
        st.subheader("👤 사용자 정보")
        username = st.text_input("사용자명", key="username_input")
        email = st.text_input("이메일 (선택사항)", key="email_input")
        
        if st.button("사용자 생성/로그인", key="user_login_btn"):
            if username:
                med_db = st.session_state.get(SESSION_KEYS["med_db"])
                if med_db:
                    user = med_db.get_or_create_user(username, email)
                    if user:
                        st.session_state[SESSION_KEYS["current_user"]] = user
                        st.success(f"환영합니다, {username}님!")
                        st.rerun()
                    else:
                        st.error("사용자 생성/로그인에 실패했습니다.")
                else:
                    st.error("데이터베이스가 연결되지 않았습니다.")
            else:
                st.error("사용자명을 입력해주세요.")
        
        # 현재 사용자 정보 표시
        current_user = st.session_state.get(SESSION_KEYS["current_user"])
        if current_user:
            st.success(f"로그인됨: {current_user['name']}")
            if st.button("로그아웃", key="logout_btn"):
                st.session_state[SESSION_KEYS["current_user"]] = None
                st.session_state[SESSION_KEYS["messages"]] = []
                st.rerun()
        
        st.divider()
        
        # 시스템 설정
        st.subheader("🔧 시스템 설정")
        
        # 테스트 모드 토글
        test_mode = st.checkbox("테스트 모드", value=st.session_state.get(SESSION_KEYS["test_mode"], False))
        if test_mode != st.session_state.get(SESSION_KEYS["test_mode"], False):
            st.session_state[SESSION_KEYS["test_mode"]] = test_mode
            st.rerun()
        
        if not test_mode:
            # Qdrant 데이터 확인 및 재인덱싱
            if st.button("Qdrant 데이터 확인", key="check_data_btn"):
                has_data = check_qdrant_data()
                if has_data:
                    st.success("✅ Qdrant에 약물 데이터가 있습니다.")
                else:
                    st.warning("⚠️ Qdrant에 약물 데이터가 없습니다.")
            
            if st.button("Qdrant 초기화", key="init_qdrant_btn"):
                if initialize_qdrant_db():
                    st.success("Qdrant 데이터베이스가 초기화되었습니다.")
                else:
                    st.error("Qdrant 초기화에 실패했습니다.")
            
            if st.button("데이터 재인덱싱", key="reindex_btn"):
                if reindex_qdrant_data():
                    st.success("데이터가 재인덱싱되었습니다.")
                else:
                    st.error("데이터 재인덱싱에 실패했습니다.")
        
        st.divider()
        
        # 예시 질문
        st.subheader("💡 예시 질문")
        example_questions = [
            "타이레놀의 효능이 뭔가요?",
            "와파린과 함께 복용하면 안 되는 약이 있나요?",
            "아세트아미노펜의 부작용은?",
            "타이레놀 복용법 알려주세요",
            "혈압약 주의사항이 궁금해요"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{hash(question)}"):
                st.session_state["example_question"] = question
                st.rerun()


def render_chat_interface():
    """채팅 인터페이스를 렌더링합니다."""
    st.subheader("💬 약물 정보 챗봇")
    
    # 메시지 히스토리 표시
    messages = st.session_state.get(SESSION_KEYS["messages"], [])
    
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        sources = message.get("sources", [])
        
        if role == "user":
            with st.chat_message("user"):
                st.write(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.write(content)
                
                # 소스 문서 표시
                if sources:
                    with st.expander("📚 참고 문서"):
                        for i, source in enumerate(sources):
                            st.write(f"**문서 {i+1}:**")
                            st.write(source.page_content[:200] + "...")
    
    # 사용자 입력
    if prompt := st.chat_input("약물에 대해 궁금한 점을 물어보세요..."):
        # 사용자 메시지 추가
        add_message("user", prompt)
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("AI가 응답을 생성하고 있습니다..."):
                # 테스트 모드인지 확인
                test_mode = st.session_state.get(SESSION_KEYS["test_mode"], False)
                
                if test_mode:
                    response = "테스트 모드입니다. 실제 AI 응답을 받으려면 테스트 모드를 해제하세요."
                    sources = []
                else:
                    # 실제 AI 응답 생성
                    from medication_tools import process_query
                    
                    conversation_id = st.session_state.get(SESSION_KEYS["conversation_id"])
                    result = process_query(prompt, conversation_id)
                    
                    if result["success"]:
                        response = result["message"]
                        sources = result.get("sources", [])
                    else:
                        response = f"오류가 발생했습니다: {result['message']}"
                        sources = []
                
                st.write(response)
                
                # 소스 문서 표시
                if sources:
                    with st.expander("📚 참고 문서"):
                        for i, source in enumerate(sources):
                            st.write(f"**문서 {i+1}:**")
                            st.write(source.page_content[:200] + "...")
                
                # AI 응답을 메시지에 추가
                add_message("assistant", response, sources)
    
    # 예시 질문 처리
    if "example_question" in st.session_state:
        example_question = st.session_state["example_question"]
        del st.session_state["example_question"]
        
        # 예시 질문을 입력에 설정
        st.rerun()


def render_medication_management():
    """복약 관리 인터페이스를 렌더링합니다."""
    st.subheader("💊 복약 관리")
    
    current_user = st.session_state.get(SESSION_KEYS["current_user"])
    if not current_user:
        st.warning("복약 관리를 사용하려면 먼저 로그인해주세요.")
        return
    
    # 약물 추가 폼
    with st.expander("➕ 새 약물 추가", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            medication_name = st.text_input("약물명", key="med_name_input")
            start_date = st.date_input("시작일", key="start_date_input")
            end_date = st.date_input("종료일", key="end_date_input")
        
        with col2:
            morning = st.checkbox("아침", key="morning_check")
            lunch = st.checkbox("점심", key="lunch_check")
            dinner = st.checkbox("저녁", key="dinner_check")
            before_meal = st.checkbox("식전", key="before_meal_check")
            after_meal = st.checkbox("식후", key="after_meal_check")
        
        if st.button("약물 추가", key="add_med_btn"):
            if medication_name:
                from medication_tools import add_medication_to_checklist
                
                result = add_medication_to_checklist(
                    medication_name=medication_name,
                    morning=morning,
                    lunch=lunch,
                    dinner=dinner,
                    before_meal=before_meal,
                    after_meal=after_meal,
                    start_date=start_date.strftime("%Y-%m-%d") if start_date else None,
                    end_date=end_date.strftime("%Y-%m-%d") if end_date else None
                )
                
                st.success(result)
                st.rerun()
            else:
                st.error("약물명을 입력해주세요.")
    
    # 현재 복약 목록
    with st.expander("📋 현재 복약 목록", expanded=True):
        from medication_tools import get_user_medications
        
        medications_info = get_user_medications()
        
        if "오류" not in medications_info and "실패" not in medications_info:
            try:
                import json
                medications = json.loads(medications_info)
                
                if medications:
                    for med in medications:
                        with st.container():
                            col1, col2, col3 = st.columns([2, 2, 1])
                            
                            with col1:
                                st.write(f"**{med['약물명']}**")
                                st.write(f"복용 시간: {med['복용 시간']}")
                            
                            with col2:
                                st.write(f"복용 시점: {med['복용 시점']}")
                                st.write(f"시작일: {med['시작일']}")
                            
                            with col3:
                                if st.button("삭제", key=f"del_{med['ID']}"):
                                    from medication_tools import delete_medication
                                    result = delete_medication(med['ID'])
                                    st.success(result)
                                    st.rerun()
                            
                            st.divider()
                else:
                    st.info("복용 중인 약물이 없습니다.")
            except:
                st.error("약물 목록을 불러올 수 없습니다.")
        else:
            st.error(medications_info)


def render_calendar_export():
    """캘린더 내보내기 인터페이스를 렌더링합니다."""
    st.subheader("📅 캘린더 내보내기")
    
    current_user = st.session_state.get(SESSION_KEYS["current_user"])
    if not current_user:
        st.warning("캘린더 내보내기를 사용하려면 먼저 로그인해주세요.")
        return
    
    # 복용 일정 생성
    with st.expander("📋 복용 일정 생성", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            drug_name = st.text_input("약물명", key="calendar_drug_name")
            unit = st.selectbox("단위", ["tablet", "mg", "ml", "patch", "drop"], key="calendar_unit")
        
        with col2:
            schedule_info = st.text_area("복용 정보", 
                                       placeholder="예: 아침 식후 1정, 저녁 식전 1정", 
                                       key="calendar_schedule")
        
        if st.button("일정 생성", key="generate_schedule_btn"):
            if drug_name and schedule_info:
                from medication_tools import create_medication_schedule
                
                schedule = create_medication_schedule(drug_name, unit, schedule_info)
                st.text_area("생성된 일정", schedule, height=200, key="generated_schedule")
                
                # 캘린더 JSON 파싱
                from calendar_utils import parse_calendar_json_block
                calendar_data = parse_calendar_json_block(schedule)
                
                if calendar_data:
                    st.success("일정이 생성되었습니다!")
                    
                    # 캘린더 내보내기 옵션
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("📅 ICS 파일로 내보내기", key="export_ics_btn"):
                            from calendar_utils import build_week_events, export_calendar
                            from config import TIMEZONE, WEEK_START_DATE
                            
                            events = build_week_events(calendar_data, WEEK_START_DATE, TIMEZONE)
                            if events:
                                file_path = export_calendar(events, "ics", f"{drug_name}_schedule")
                                st.success(f"ICS 파일이 내보내졌습니다: {file_path}")
                            else:
                                st.error("이벤트를 생성할 수 없습니다.")
                    
                    with col2:
                        if st.button("📄 JSON 파일로 내보내기", key="export_json_btn"):
                            from calendar_utils import build_week_events, export_calendar
                            from config import TIMEZONE, WEEK_START_DATE
                            
                            events = build_week_events(calendar_data, WEEK_START_DATE, TIMEZONE)
                            if events:
                                file_path = export_calendar(events, "json", f"{drug_name}_schedule")
                                st.success(f"JSON 파일이 내보내졌습니다: {file_path}")
                            else:
                                st.error("이벤트를 생성할 수 없습니다.")
                else:
                    st.error("일정을 파싱할 수 없습니다. AI 응답을 확인해주세요.")
            else:
                st.error("약물명과 복용 정보를 모두 입력해주세요.")


def initialize_system():
    """시스템을 초기화합니다."""
    # 페이지 설정
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT
    )
    
    # 세션 상태 초기화
    initialize_session_state()
    
    # 데이터베이스 연결
    if st.session_state.get(SESSION_KEYS["med_db"]) is None:
        med_db = MedicationDatabase()
        if med_db.conn:  # 연결 성공한 경우에만 저장
            st.session_state[SESSION_KEYS["med_db"]] = med_db
    
    # AI 모델 초기화 (테스트 모드가 아닌 경우)
    test_mode = st.session_state.get(SESSION_KEYS["test_mode"], False)
    
    if not test_mode:
        # OpenAI 클라이언트 초기화
        if st.session_state.get("llm") is None:
            llm = initialize_openai()
            if llm:
                st.session_state["llm"] = llm
        
        # Qdrant 클라이언트 초기화
        if st.session_state.get(SESSION_KEYS["qdrant_client"]) is None:
            qdrant_client = initialize_qdrant()
            if qdrant_client:
                st.session_state[SESSION_KEYS["qdrant_client"]] = qdrant_client
        
        # 임베딩 모델 초기화
        if st.session_state.get("embeddings") is None:
            embeddings = initialize_embeddings()
            if embeddings:
                st.session_state["embeddings"] = embeddings
        
        # 에이전트 초기화
        if st.session_state.get(SESSION_KEYS["agent"]) is None:
            agent = create_agent()
            if agent:
                st.session_state[SESSION_KEYS["agent"]] = agent
        
        # QA 체인 초기화
        if st.session_state.get(SESSION_KEYS["qa_chain"]) is None:
            qa_chain = initialize_qa_chain()
            if qa_chain:
                st.session_state[SESSION_KEYS["qa_chain"]] = qa_chain
