"""
💊 Medication Agent - 약물 관련 도구
약물 정보 검색, 복약 일정 생성, 체크리스트 관리 등의 기능을 제공합니다.
"""

import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import streamlit as st

from medication_db import MedicationDatabase
from calendar_utils import parse_calendar_json_block, build_week_events, export_calendar
from config import TIMEZONE, WEEK_START_DATE, SUCCESS_MESSAGES, ERROR_MESSAGES


def get_chat_history() -> str:
    """
    현재 세션의 채팅 히스토리를 반환합니다.
    
    Returns:
        str: 채팅 히스토리 문자열
    """
    messages = st.session_state.get("messages", [])
    if not messages:
        return "대화 기록이 없습니다."
    
    history = []
    for msg in messages[-10:]:  # 최근 10개 메시지만
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        if role == "user":
            history.append(f"사용자: {content}")
        elif role == "assistant":
            history.append(f"AI: {content}")
    
    return "\n".join(history)


def create_medication_schedule(drug_name: str, unit: str, schedule_info: str) -> str:
    """
    복약 일정을 생성합니다.
    
    Args:
        drug_name (str): 약물명
        unit (str): 단위 (tablet, mg, ml 등)
        schedule_info (str): 복용 정보
        
    Returns:
        str: 생성된 일정
    """
    try:
        # OpenAI 클라이언트 가져오기
        llm = st.session_state.get("llm")
        if not llm:
            return "OpenAI 클라이언트가 초기화되지 않았습니다."
        
        # 프롬프트 생성
        prompt = f"""
        사용자가 복용하려는 약물의 일정을 생성해주세요.
        다음 정보를 바탕으로 7일간의 복용 일정을 만들어주세요:
        
        약물명: {drug_name}
        단위: {unit}
        복용 정보: {schedule_info}
        
        응답은 다음 JSON 형식으로 제공해주세요:
        [[CALENDAR_JSON]]
        {{
          "drug_name": "{drug_name}",
          "unit": "{unit}",
          "per_slot": {{
            "morning_before": 0,
            "morning_after": 0,
            "lunch_before": 0,
            "lunch_after": 0,
            "evening_before": 0,
            "evening_after": 0
          }},
          "notes": "추가 참고사항"
        }}
        [[/CALENDAR_JSON]]
        
        각 슬롯의 숫자는 해당 시간대에 복용할 약물의 양입니다.
        """
        
        # AI 응답 생성
        response = llm.invoke(prompt)
        
        return response.content
        
    except Exception as e:
        return f"일정 생성 중 오류가 발생했습니다: {e}"


def add_medication_to_checklist(medication_name: str, morning: bool = False, 
                               lunch: bool = False, dinner: bool = False, 
                               before_meal: bool = False, after_meal: bool = False, 
                               start_date: str = None, end_date: str = None) -> str:
    """
    체크리스트에 약물을 추가합니다.
    
    Args:
        medication_name (str): 약물명
        morning (bool): 아침 복용 여부
        lunch (bool): 점심 복용 여부
        dinner (bool): 저녁 복용 여부
        before_meal (bool): 식전 복용 여부
        after_meal (bool): 식후 복용 여부
        start_date (str): 시작 날짜
        end_date (str): 종료 날짜
        
    Returns:
        str: 추가 결과 메시지
    """
    try:
        # 데이터베이스 연결 확인
        med_db = st.session_state.get("med_db")
        if not med_db:
            return "데이터베이스가 연결되지 않았습니다."
        
        # 현재 사용자 확인
        current_user = st.session_state.get("current_user")
        if not current_user:
            return "사용자 정보를 찾을 수 없습니다."
        
        # 날짜 파싱
        start_date_obj = None
        end_date_obj = None
        
        if start_date:
            try:
                start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
            except ValueError:
                return "시작 날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요."
        
        if end_date:
            try:
                end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
            except ValueError:
                return "종료 날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요."
        
        # 약물 데이터 구성
        medication_data = {
            "medication_name": medication_name,
            "morning": morning,
            "lunch": lunch,
            "dinner": dinner,
            "before_meal": before_meal,
            "after_meal": after_meal,
            "start_date": start_date_obj or date.today(),
            "end_date": end_date_obj
        }
        
        # 데이터베이스에 추가
        result = med_db.add_medication(current_user["id"], medication_data)
        
        if result:
            return SUCCESS_MESSAGES["medication_added"]
        else:
            return "약물 추가에 실패했습니다."
        
    except Exception as e:
        return f"약물 추가 중 오류가 발생했습니다: {e}"


def get_user_medications() -> str:
    """
    사용자의 복약 체크리스트를 반환합니다.
    
    Returns:
        str: 복약 체크리스트 정보
    """
    try:
        # 데이터베이스 연결 확인
        med_db = st.session_state.get("med_db")
        if not med_db:
            return "데이터베이스가 연결되지 않았습니다."
        
        # 현재 사용자 확인
        current_user = st.session_state.get("current_user")
        if not current_user:
            return "사용자 정보를 찾을 수 없습니다."
        
        # 사용자의 약물 목록 조회
        medications = med_db.get_user_medications(current_user["id"])
        
        if not medications:
            return "복용 중인 약물이 없습니다."
        
        # 결과 포맷팅
        result = []
        for med in medications:
            schedule = []
            if med.get("morning"):
                schedule.append("아침")
            if med.get("lunch"):
                schedule.append("점심")
            if med.get("dinner"):
                schedule.append("저녁")
            
            timing = []
            if med.get("before_meal"):
                timing.append("식전")
            if med.get("after_meal"):
                timing.append("식후")
            
            schedule_str = ", ".join(schedule) if schedule else "설정 안됨"
            timing_str = ", ".join(timing) if timing else "설정 안됨"
            
            result.append({
                "ID": med["id"],
                "약물명": med["medication_name"],
                "복용 시간": schedule_str,
                "복용 시점": timing_str,
                "시작일": med["start_date"].strftime("%Y-%m-%d") if med["start_date"] else "설정 안됨",
                "종료일": med["end_date"].strftime("%Y-%m-%d") if med["end_date"] else "설정 안됨"
            })
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return f"약물 목록 조회 중 오류가 발생했습니다: {e}"


def update_medication(medication_id: int, medication_name: str = None, 
                     morning: bool = None, lunch: bool = None, dinner: bool = None, 
                     before_meal: bool = None, after_meal: bool = None, 
                     start_date: str = None, end_date: str = None) -> str:
    """
    복약 정보를 업데이트합니다.
    
    Args:
        medication_id (int): 약물 ID
        medication_name (str): 약물명
        morning (bool): 아침 복용 여부
        lunch (bool): 점심 복용 여부
        dinner (bool): 저녁 복용 여부
        before_meal (bool): 식전 복용 여부
        after_meal (bool): 식후 복용 여부
        start_date (str): 시작 날짜
        end_date (str): 종료 날짜
        
    Returns:
        str: 업데이트 결과 메시지
    """
    try:
        # 데이터베이스 연결 확인
        med_db = st.session_state.get("med_db")
        if not med_db:
            return "데이터베이스가 연결되지 않았습니다."
        
        # 현재 사용자 확인
        current_user = st.session_state.get("current_user")
        if not current_user:
            return "사용자 정보를 찾을 수 없습니다."
        
        # 업데이트할 데이터 구성
        update_data = {}
        
        if medication_name is not None:
            update_data["medication_name"] = medication_name
        if morning is not None:
            update_data["morning"] = morning
        if lunch is not None:
            update_data["lunch"] = lunch
        if dinner is not None:
            update_data["dinner"] = dinner
        if before_meal is not None:
            update_data["before_meal"] = before_meal
        if after_meal is not None:
            update_data["after_meal"] = after_meal
        
        # 날짜 파싱
        if start_date:
            try:
                update_data["start_date"] = datetime.strptime(start_date, "%Y-%m-%d").date()
            except ValueError:
                return "시작 날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요."
        
        if end_date:
            try:
                update_data["end_date"] = datetime.strptime(end_date, "%Y-%m-%d").date()
            except ValueError:
                return "종료 날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요."
        
        if not update_data:
            return "업데이트할 내용이 없습니다."
        
        # 데이터베이스 업데이트
        result = med_db.update_medication(medication_id, update_data)
        
        if result:
            return SUCCESS_MESSAGES["medication_updated"]
        else:
            return "약물 정보 업데이트에 실패했습니다."
        
    except Exception as e:
        return f"약물 정보 업데이트 중 오류가 발생했습니다: {e}"


def delete_medication(medication_id: int) -> str:
    """
    복약 정보를 삭제합니다.
    
    Args:
        medication_id (int): 약물 ID
        
    Returns:
        str: 삭제 결과 메시지
    """
    try:
        # 데이터베이스 연결 확인
        med_db = st.session_state.get("med_db")
        if not med_db:
            return "데이터베이스가 연결되지 않았습니다."
        
        # 현재 사용자 확인
        current_user = st.session_state.get("current_user")
        if not current_user:
            return "사용자 정보를 찾을 수 없습니다."
        
        # 데이터베이스에서 삭제
        result = med_db.delete_medication(medication_id)
        
        if result:
            return SUCCESS_MESSAGES["medication_deleted"]
        else:
            return "약물 삭제에 실패했습니다."
        
    except Exception as e:
        return f"약물 삭제 중 오류가 발생했습니다: {e}"


def parse_medication_info(user_input: str) -> Optional[Dict[str, Any]]:
    """
    사용자 입력에서 약물 정보를 파싱합니다.
    
    Args:
        user_input (str): 사용자 입력
        
    Returns:
        Optional[Dict[str, Any]]: 파싱된 약물 정보 또는 None
    """
    try:
        # 약물명 추출 패턴들
        patterns = [
            r"([가-힣a-zA-Z0-9]+)\s*(\d+)\s*(정|mg|ml|patch|drop)",
            r"([가-힣a-zA-Z0-9]+)\s*(정|mg|ml|patch|drop)\s*(\d+)",
            r"([가-힣a-zA-Z0-9]+)\s*(\d+)\s*(알|개|정)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input)
            if match:
                drug_name = match.group(1)
                amount = int(match.group(2))
                unit = match.group(3)
                
                # 단위 정규화
                unit_mapping = {
                    "정": "tablet", "알": "tablet", "개": "tablet",
                    "mg": "mg", "ml": "ml", "patch": "patch", "drop": "drop"
                }
                normalized_unit = unit_mapping.get(unit, unit)
                
                # 복용 시간 추출
                morning = "아침" in user_input or "morning" in user_input.lower()
                lunch = "점심" in user_input or "lunch" in user_input.lower()
                dinner = "저녁" in user_input or "dinner" in user_input.lower()
                before_meal = "식전" in user_input or "before" in user_input.lower()
                after_meal = "식후" in user_input or "after" in user_input.lower()
                
                return {
                    "drug_name": drug_name,
                    "amount": amount,
                    "unit": normalized_unit,
                    "morning": morning,
                    "lunch": lunch,
                    "dinner": dinner,
                    "before_meal": before_meal,
                    "after_meal": after_meal
                }
        
        return None
        
    except Exception:
        return None


def classify_user_input(user_input: str) -> Dict:
    """
    사용자 입력을 분류합니다.
    
    Args:
        user_input (str): 사용자 입력
        
    Returns:
        Dict: 분류 결과
    """
    user_input_lower = user_input.lower()
    
    # 약물 정보 추가 관련 키워드
    add_keywords = ["추가", "등록", "넣어", "add", "register", "input"]
    add_medication = any(keyword in user_input_lower for keyword in add_keywords)
    
    # 약물 정보 조회 관련 키워드
    view_keywords = ["보여", "조회", "목록", "체크리스트", "show", "list", "view"]
    view_medication = any(keyword in user_input_lower for keyword in view_keywords)
    
    # 약물 정보 수정 관련 키워드
    edit_keywords = ["수정", "변경", "바꿔", "edit", "modify", "change", "update"]
    edit_medication = any(keyword in user_input_lower for keyword in edit_keywords)
    
    # 약물 정보 삭제 관련 키워드
    delete_keywords = ["삭제", "제거", "지워", "delete", "remove"]
    delete_medication = any(keyword in user_input_lower for keyword in delete_keywords)
    
    # 약물 질문 관련 키워드
    question_keywords = ["뭐", "무엇", "어떻게", "언제", "어디서", "왜", "how", "what", "when", "where", "why"]
    is_question = any(keyword in user_input_lower for keyword in question_keywords)
    
    # 복용 일정 관련 키워드
    schedule_keywords = ["일정", "스케줄", "복용", "복용법", "schedule", "dosage", "medication"]
    is_schedule = any(keyword in user_input_lower for keyword in schedule_keywords)
    
    return {
        "add_medication": add_medication,
        "view_medication": view_medication,
        "edit_medication": edit_medication,
        "delete_medication": delete_medication,
        "is_question": is_question,
        "is_schedule": is_schedule,
        "input_type": "medication_management" if any([add_medication, view_medication, edit_medication, delete_medication]) else "question"
    }


def process_medication_info(user_input: str, classification: Dict, 
                          med_db: MedicationDatabase, current_user: Dict) -> Dict:
    """
    약물 정보 관련 입력을 처리합니다.
    
    Args:
        user_input (str): 사용자 입력
        classification (Dict): 분류 결과
        med_db (MedicationDatabase): 약물 데이터베이스
        current_user (Dict): 현재 사용자
        
    Returns:
        Dict: 처리 결과
    """
    try:
        if classification["add_medication"]:
            # 약물 정보 파싱
            med_info = parse_medication_info(user_input)
            if not med_info:
                return {
                    "success": False,
                    "message": "약물 정보를 파싱할 수 없습니다. 형식을 확인해주세요.",
                    "action": "parse_error"
                }
            
            # 체크리스트에 추가
            result = add_medication_to_checklist(
                medication_name=med_info["drug_name"],
                morning=med_info["morning"],
                lunch=med_info["lunch"],
                dinner=med_info["dinner"],
                before_meal=med_info["before_meal"],
                after_meal=med_info["after_meal"]
            )
            
            return {
                "success": "실패" not in result,
                "message": result,
                "action": "add_medication"
            }
        
        elif classification["view_medication"]:
            # 복약 체크리스트 조회
            result = get_user_medications()
            
            return {
                "success": "오류" not in result,
                "message": result,
                "action": "view_medication"
            }
        
        elif classification["edit_medication"]:
            # 수정 기능은 향후 구현
            return {
                "success": False,
                "message": "약물 정보 수정 기능은 아직 구현되지 않았습니다.",
                "action": "edit_medication"
            }
        
        elif classification["delete_medication"]:
            # 삭제 기능은 향후 구현
            return {
                "success": False,
                "message": "약물 정보 삭제 기능은 아직 구현되지 않았습니다.",
                "action": "delete_medication"
            }
        
        else:
            return {
                "success": False,
                "message": "처리할 수 없는 입력입니다.",
                "action": "unknown"
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"처리 중 오류가 발생했습니다: {e}",
            "action": "error"
        }


def process_medication_question(user_input: str, conversation_id: str) -> Dict:
    """
    약물 관련 질문을 처리합니다.
    
    Args:
        user_input (str): 사용자 입력
        conversation_id (str): 대화 ID
        
    Returns:
        Dict: 처리 결과
    """
    try:
        # QA 체인 가져오기
        qa_chain = st.session_state.get("qa_chain")
        if not qa_chain:
            return {
                "success": False,
                "message": "AI 모델이 초기화되지 않았습니다.",
                "action": "ai_not_initialized"
            }
        
        # 질문 처리
        result = qa_chain({"query": user_input})
        ai_response = result.get("result", "응답을 생성할 수 없습니다.")
        sources = result.get("source_documents", [])
        
        return {
            "success": True,
            "message": ai_response,
            "sources": sources,
            "action": "question_answered"
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"질문 처리 중 오류가 발생했습니다: {e}",
            "action": "error"
        }


def smart_agent_response(user_input: str, med_db: MedicationDatabase, 
                        current_user: Dict, conversation_id: str) -> Dict:
    """
    스마트 에이전트를 사용하여 응답을 생성합니다.
    
    Args:
        user_input (str): 사용자 입력
        med_db (MedicationDatabase): 약물 데이터베이스
        current_user (Dict): 현재 사용자
        conversation_id (str): 대화 ID
        
    Returns:
        Dict: 응답 결과
    """
    try:
        # 에이전트 가져오기
        agent = st.session_state.get("agent")
        if not agent:
            return {
                "success": False,
                "message": "AI 에이전트가 초기화되지 않았습니다.",
                "action": "agent_not_initialized"
            }
        
        # 에이전트 실행
        result = agent.invoke({
            "input": user_input,
            "chat_history": get_chat_history()
        })
        
        return {
            "success": True,
            "message": result.get("output", "응답을 생성할 수 없습니다."),
            "action": "agent_response"
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"에이전트 실행 중 오류가 발생했습니다: {e}",
            "action": "error"
        }


def process_query(user_input: str, conversation_id: str = None) -> Dict:
    """
    사용자 쿼리를 처리합니다.
    
    Args:
        user_input (str): 사용자 입력
        conversation_id (str): 대화 ID
        
    Returns:
        Dict: 처리 결과
    """
    try:
        # 데이터베이스 및 사용자 정보 확인
        med_db = st.session_state.get("med_db")
        current_user = st.session_state.get("current_user")
        
        if not med_db or not current_user:
            return {
                "success": False,
                "message": "시스템이 초기화되지 않았습니다.",
                "action": "system_not_initialized"
            }
        
        # 입력 분류
        classification = classify_user_input(user_input)
        
        # 약물 관리 관련 입력 처리
        if classification["input_type"] == "medication_management":
            return process_medication_info(user_input, classification, med_db, current_user)
        
        # 약물 질문 처리
        elif classification["is_question"]:
            return process_medication_question(user_input, conversation_id)
        
        # 스마트 에이전트 처리
        else:
            return smart_agent_response(user_input, med_db, current_user, conversation_id)
            
    except Exception as e:
        return {
            "success": False,
            "message": f"쿼리 처리 중 오류가 발생했습니다: {e}",
            "action": "error"
        }
