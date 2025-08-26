#!/usr/bin/env python3
"""
💊 Medication Agent - Tool 기반 버전
LangChain Tool을 활용한 약물 정보 챗봇
"""

import os
import json
from typing import List, Dict, Any, TypedDict, Optional
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from qdrant_client import QdrantClient

# 복약 데이터베이스 import
from medication_db import MedicationDatabase

# 환경 변수 로드
load_dotenv()

# ========== 캘린더 출력 유틸 ==========

TIMEZONE = os.getenv("APP_TZ", "Asia/Seoul")          # 기본 타임존
EXPORT_DIR = os.getenv("CAL_EXPORT_DIR", "./_calendar_exports")  # 저장 폴더
DEFAULT_MEAL_TIMES = {                                   # 기본 식사 시각
    "morning": time(8, 0),   # 아침 08:00
    "lunch":   time(12, 30), # 점심 12:30
    "evening": time(19, 0),  # 저녁 19:00
}
PRE_MEAL_OFFSET_MIN = 30   # 식전 30분
POST_MEAL_OFFSET_MIN = 30  # 식후 30분
WEEK_START_DATE = date(2025, 8, 24)  # 요구: 2025-08-24(일)부터 1주
WEEK_DAYS_KO = ["월", "화", "수", "목", "금", "토", "일"]  # datetime.weekday() 대응

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _parse_calendar_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    AI 응답 안의 [[CALENDAR_JSON]] ... [[/CALENDAR_JSON]] 블록을 찾아 JSON 파싱.
    """
    import re
    m = re.search(r"\[\[CALENDAR_JSON\]\]\s*(\{.*?\})\s*\[\[/CALENDAR_JSON\]\]", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

def _slot_to_datetime(d: date, slot_key: str, tz: str) -> datetime:
    # slot_key: morning_before | morning_after | lunch_before | lunch_after | evening_before | evening_after
    meal, when = slot_key.split("_")
    base_t = DEFAULT_MEAL_TIMES[meal]
    base_dt = datetime.combine(d, base_t, tzinfo=ZoneInfo(tz))
    return base_dt - timedelta(minutes=PRE_MEAL_OFFSET_MIN) if when == "before" else base_dt + timedelta(minutes=POST_MEAL_OFFSET_MIN)

def _build_week_events(plan: Dict[str, Any], start_date: date, tz: str) -> List[Dict[str, Any]]:
    """
    plan = {
      "drug_name": str,
      "unit": str,  # tablet|mg|ml|patch|drop
      "per_slot": { six keys...: int },
      "notes": str (optional)
    }
    """
    events = []
    per_slot: Dict[str, Any] = plan.get("per_slot", {})
    drug = plan.get("drug_name", "약물")
    unit = plan.get("unit", "tablet")
    notes = plan.get("notes", "")

    for day_offset in range(7):
        d = start_date + timedelta(days=day_offset)
        weekday_ko = WEEK_DAYS_KO[d.weekday()]
        for slot_key in ["morning_before","morning_after","lunch_before","lunch_after","evening_before","evening_after"]:
            val = per_slot.get(slot_key, 0)
            try:
                count = int(val)
            except Exception:
                continue
            if count <= 0:
                continue

            start_dt = _slot_to_datetime(d, slot_key, tz)
            end_dt = start_dt + timedelta(minutes=10)

            label_map = {"morning":"아침","lunch":"점심","evening":"저녁","before":"식전","after":"식후"}
            meal, when = slot_key.split("_")
            human_slot = f"{label_map[meal]} {label_map[when]}"

            events.append({
                "uid": f"medication-schedule@medication-agent",
                "drug_name": drug,
                "unit": unit,
                "dose": count,
                "slot": slot_key,
                "slot_ko": human_slot,
                "date": d.isoformat(),
                "weekday_ko": weekday_ko,
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
                "notes": notes
            })
    return events

def _export_json(events: List[Dict[str, Any]], out_path: str):
    payload = {
        "generated_at": datetime.now(tz=ZoneInfo(TIMEZONE)).isoformat(),
        "timezone": TIMEZONE,
        "week_start_date": WEEK_START_DATE.isoformat(),
        "events": events
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _escape_ics(text: str) -> str:
    return text.replace("\\","\\\\").replace(",","\\,").replace(";","\\;").replace("\n","\\n")

def _dt_local(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%S")

def _export_ics(events: List[Dict[str, Any]], out_path: str, tz: str):
    lines = ["BEGIN:VCALENDAR","PRODID:-//Medication Agent//calendar export//EN","VERSION:2.0",f"X-WR-TIMEZONE:{tz}"]
    now_utc = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    for ev in events:
        start_dt = datetime.fromisoformat(ev["start"])
        end_dt = datetime.fromisoformat(ev["end"])
        summary = f"💊 {ev['drug_name']} x{ev['dose']} ({ev['slot_ko']})"
        desc = ev.get("notes","")

        lines += [
            "BEGIN:VEVENT",
            f"UID:medication-schedule@medication-agent",
            f"DTSTAMP:{now_utc}",
            f"DTSTART;TZID={tz}:{_dt_local(start_dt)}",
            f"DTEND;TZID={tz}:{_dt_local(end_dt)}",
            f"SUMMARY:{_escape_ics(summary)}",
        ]
        if desc:
            lines.append(f"DESCRIPTION:{_escape_ics(desc)}")
        lines.append("END:VEVENT")

    lines.append("END:VCALENDAR")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# 페이지 설정
st.set_page_config(
    page_title="💊 Medication Agent",
    page_icon="💊",
    layout="wide"
)

# Tool 정의
@tool
def search_drug_info(query: str) -> str:
    """
    약물 정보를 검색하는 도구입니다.
    
    Args:
        query: 검색할 약물명이나 질문 (예: "타이레놀", "두통에 좋은 약", "아세트아미노펜 부작용")
    
    Returns:
        검색된 약물 정보를 JSON 형태로 반환합니다.
    """
    try:
        # Qdrant 클라이언트
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        # Embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 검색 실행
        search_results = qdrant_client.search(
            collection_name="product_sections",
            query_vector=embeddings.embed_query(query),
            limit=5,
            with_payload=True
        )
        
        # 검색 결과 품질 확인 (점수가 너무 낮으면 필터링)
        quality_results = []
        for result in search_results:
            if result.score > 0.3:  # 점수가 0.3 이상인 결과만 사용
                quality_results.append(result)
        
        # 품질 좋은 결과가 없으면 원본 결과 사용
        if not quality_results:
            quality_results = search_results[:3]
        else:
            quality_results = quality_results[:3]
        
        # 결과를 JSON 형태로 변환
        results = []
        for result in quality_results:
            results.append({
                "item_name": result.payload.get('item_name', ''),
                "section": result.payload.get('section', ''),
                "text": result.payload.get('text', ''),
                "score": f"{result.score:.3f}"
            })
        
        if not results:
            return json.dumps({
                "status": "no_results",
                "message": "관련 정보를 찾을 수 없습니다.",
                "results": []
            }, ensure_ascii=False)
        
        return json.dumps({
            "status": "success",
            "message": f"{len(results)}개의 관련 정보를 찾았습니다.",
            "results": results
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"검색 중 오류가 발생했습니다: {str(e)}",
            "results": []
        }, ensure_ascii=False)

@tool
def get_chat_history() -> str:
    """
    이전 대화 히스토리를 가져오는 도구입니다.
    
    Returns:
        이전 대화 내용을 JSON 형태로 반환합니다.
    """
    try:
        chat_history = st.session_state.get("chat_history", [])
        
        if not chat_history:
            return json.dumps({
                "status": "no_history",
                "message": "이전 대화 기록이 없습니다.",
                "history": []
            }, ensure_ascii=False)
        
        # 최근 4개 대화만 반환
        recent_history = chat_history[-4:]
        
        return json.dumps({
            "status": "success",
            "message": f"최근 {len(recent_history)}개의 대화를 찾았습니다.",
            "history": recent_history
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"히스토리 조회 중 오류가 발생했습니다: {str(e)}",
            "history": []
        }, ensure_ascii=False)

@tool
def create_medication_schedule(drug_name: str, unit: str, schedule_info: str) -> str:
    """
    약물 복용 스케줄을 생성하고 캘린더 파일(JSON/ICS)을 만드는 도구입니다.
    
    Args:
        drug_name: 약물명 (예: "타이레놀 500mg")
        unit: 단위 (예: "tablet", "mg", "ml", "patch", "drop")
        schedule_info: 복용 스케줄 정보 (예: "매일 아침 식후 1정, 저녁 식후 1정")
    
    Returns:
        생성된 파일 경로와 스케줄 정보를 JSON 형태로 반환합니다.
    """
    try:
        # 스케줄 정보를 파싱하여 per_slot 구조로 변환
        per_slot = {
            "morning_before": 0,
            "morning_after": 0,
            "lunch_before": 0,
            "lunch_after": 0,
            "evening_before": 0,
            "evening_after": 0
        }
        
        # 스케줄 정보 파싱 (간단한 키워드 매칭)
        schedule_lower = schedule_info.lower()
        
        # 아침
        if "아침" in schedule_lower or "morning" in schedule_lower:
            if "식전" in schedule_lower or "before" in schedule_lower:
                per_slot["morning_before"] = 1
            else:
                per_slot["morning_after"] = 1
        
        # 점심
        if "점심" in schedule_lower or "lunch" in schedule_lower:
            if "식전" in schedule_lower or "before" in schedule_lower:
                per_slot["lunch_before"] = 1
            else:
                per_slot["lunch_after"] = 1
        
        # 저녁
        if "저녁" in schedule_lower or "evening" in schedule_lower:
            if "식전" in schedule_lower or "before" in schedule_lower:
                per_slot["evening_before"] = 1
            else:
                per_slot["evening_after"] = 1
        
        # 복용량 추출 (기본값: 1)
        dose = 1
        import re
        dose_match = re.search(r'(\d+)\s*(정|mg|ml|patch|drop)', schedule_info)
        if dose_match:
            dose = int(dose_match.group(1))
        
        # 모든 슬롯에 복용량 적용
        for key in per_slot:
            if per_slot[key] > 0:
                per_slot[key] = dose
        
        # 스케줄 계획 생성
        plan = {
            "drug_name": drug_name,
            "unit": unit,
            "per_slot": per_slot,
            "notes": f"복용 스케줄: {schedule_info}"
        }
        
        # 파일 생성
        _ensure_dir(EXPORT_DIR)
        conversation_id = st.session_state.get("conversation_id", f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        base = f"med_schedule_{conversation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        json_path = os.path.join(EXPORT_DIR, base + ".json")
        ics_path = os.path.join(EXPORT_DIR, base + ".ics")
        
        # 이벤트 생성 및 파일 출력
        events = _build_week_events(plan, WEEK_START_DATE, TIMEZONE)
        _export_json(events, json_path)
        _export_ics(events, ics_path, TIMEZONE)
        
        return json.dumps({
            "status": "success",
            "message": "약물 복용 스케줄이 생성되었습니다.",
            "files": {
                "json": os.path.abspath(json_path),
                "ics": os.path.abspath(ics_path)
            },
            "schedule": {
                "drug_name": drug_name,
                "unit": unit,
                "per_slot": per_slot,
                "total_events": len(events),
                "week_start": WEEK_START_DATE.isoformat(),
                "timezone": TIMEZONE
            }
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"스케줄 생성 중 오류가 발생했습니다: {str(e)}",
            "files": {},
            "schedule": {}
        }, ensure_ascii=False)

@tool
def add_medication_to_checklist(medication_name: str, morning: bool = False, lunch: bool = False, dinner: bool = False, before_meal: bool = False, after_meal: bool = False, start_date: str = None, end_date: str = None) -> str:
    """
    사용자의 복약 정보를 체크리스트에 추가하는 도구입니다.
    
    Args:
        medication_name: 약물명 (필수)
        morning: 아침 복용 여부 (기본값: False)
        lunch: 점심 복용 여부 (기본값: False)
        dinner: 저녁 복용 여부 (기본값: False)
        before_meal: 식전 복용 여부 (기본값: False)
        after_meal: 식후 복용 여부 (기본값: False)
        start_date: 시작일 (YYYY-MM-DD 형식, 기본값: 오늘)
        end_date: 종료일 (YYYY-MM-DD 형식, 기본값: None)
    
    Returns:
        추가 결과를 JSON 형태로 반환합니다.
    """
    try:
        # 현재 사용자 확인
        current_user = st.session_state.get("current_user")
        if not current_user:
            return json.dumps({
                "status": "error",
                "message": "로그인이 필요합니다.",
                "medication": None
            }, ensure_ascii=False)
        
        # 시작일이 없으면 오늘 날짜로 설정
        if not start_date:
            start_date = datetime.now().date().isoformat()
        
        # 복약 정보 구성
        medication_data = {
            'medication_name': medication_name,
            'morning': morning,
            'lunch': lunch,
            'dinner': dinner,
            'before_meal': before_meal,
            'after_meal': after_meal,
            'start_date': start_date,
            'end_date': end_date
        }
        
        # 데이터베이스에 추가
        from medication_db import MedicationDatabase
        med_db = MedicationDatabase()
        
        if med_db.add_medication(current_user['id'], medication_data):
            return json.dumps({
                "status": "success",
                "message": f"✅ '{medication_name}' 복약 정보가 체크리스트에 추가되었습니다!",
                "medication": medication_data
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "status": "error",
                "message": "복약 정보 추가에 실패했습니다.",
                "medication": None
            }, ensure_ascii=False)
            
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"복약 정보 추가 중 오류가 발생했습니다: {str(e)}",
            "medication": None
        }, ensure_ascii=False)

@tool
def get_user_medications() -> str:
    """
    현재 사용자의 복약 체크리스트를 조회하는 도구입니다.
    
    Returns:
        사용자의 복약 정보 목록을 JSON 형태로 반환합니다.
    """
    try:
        # 현재 사용자 확인
        current_user = st.session_state.get("current_user")
        if not current_user:
            return json.dumps({
                "status": "error",
                "message": "로그인이 필요합니다.",
                "medications": []
            }, ensure_ascii=False)
        
        # 데이터베이스에서 복약 정보 조회
        from medication_db import MedicationDatabase
        med_db = MedicationDatabase()
        
        medications = med_db.get_user_medications(current_user['id'])
        
        return json.dumps({
            "status": "success",
            "message": f"총 {len(medications)}개의 복약 정보를 찾았습니다.",
            "medications": medications
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"복약 정보 조회 중 오류가 발생했습니다: {str(e)}",
            "medications": []
        }, ensure_ascii=False)

@tool
def update_medication(medication_id: int, medication_name: str = None, morning: bool = None, lunch: bool = None, dinner: bool = None, before_meal: bool = None, after_meal: bool = None, start_date: str = None, end_date: str = None) -> str:
    """
    복약 정보를 수정하는 도구입니다.
    
    Args:
        medication_id: 수정할 복약 정보의 ID (필수)
        medication_name: 약물명
        morning: 아침 복용 여부
        lunch: 점심 복용 여부
        dinner: 저녁 복용 여부
        before_meal: 식전 복용 여부
        after_meal: 식후 복용 여부
        start_date: 시작일 (YYYY-MM-DD 형식)
        end_date: 종료일 (YYYY-MM-DD 형식)
    
    Returns:
        수정 결과를 JSON 형태로 반환합니다.
    """
    try:
        # 현재 사용자 확인
        current_user = st.session_state.get("current_user")
        if not current_user:
            return json.dumps({
                "status": "error",
                "message": "로그인이 필요합니다."
            }, ensure_ascii=False)
        
        # 업데이트할 데이터 구성
        update_data = {}
        if medication_name is not None:
            update_data['medication_name'] = medication_name
        if morning is not None:
            update_data['morning'] = morning
        if lunch is not None:
            update_data['lunch'] = lunch
        if dinner is not None:
            update_data['dinner'] = dinner
        if before_meal is not None:
            update_data['before_meal'] = before_meal
        if after_meal is not None:
            update_data['after_meal'] = after_meal
        if start_date is not None:
            update_data['start_date'] = start_date
        if end_date is not None:
            update_data['end_date'] = end_date
        
        if not update_data:
            return json.dumps({
                "status": "error",
                "message": "수정할 정보가 없습니다."
            }, ensure_ascii=False)
        
        # 데이터베이스에서 수정
        from medication_db import MedicationDatabase
        med_db = MedicationDatabase()
        
        if med_db.update_medication(medication_id, update_data):
            return json.dumps({
                "status": "success",
                "message": "✅ 복약 정보가 수정되었습니다!"
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "status": "error",
                "message": "복약 정보 수정에 실패했습니다."
            }, ensure_ascii=False)
            
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"복약 정보 수정 중 오류가 발생했습니다: {str(e)}"
        }, ensure_ascii=False)

@tool
def delete_medication(medication_id: int) -> str:
    """
    복약 정보를 삭제하는 도구입니다.
    
    Args:
        medication_id: 삭제할 복약 정보의 ID (필수)
    
    Returns:
        삭제 결과를 JSON 형태로 반환합니다.
    """
    try:
        # 현재 사용자 확인
        current_user = st.session_state.get("current_user")
        if not current_user:
            return json.dumps({
                "status": "error",
                "message": "로그인이 필요합니다."
            }, ensure_ascii=False)
        
        # 데이터베이스에서 삭제
        from medication_db import MedicationDatabase
        med_db = MedicationDatabase()
        
        if med_db.delete_medication(medication_id):
            return json.dumps({
                "status": "success",
                "message": "✅ 복약 정보가 삭제되었습니다!"
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "status": "error",
                "message": "복약 정보 삭제에 실패했습니다."
            }, ensure_ascii=False)
            
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"복약 정보 삭제 중 오류가 발생했습니다: {str(e)}"
        }, ensure_ascii=False)

# Agent 생성 함수
def create_agent():
    """LangChain Agent 생성"""
    # LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Tools
    tools = [
        search_drug_info, 
        get_chat_history, 
        create_medication_schedule,
        add_medication_to_checklist,
        get_user_medications,
        update_medication,
        delete_medication
    ]
    
    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 한국의 약물 정보 전문가입니다. 식약처 공공데이터를 바탕으로 정확하고 도움이 되는 답변을 제공해주세요.

매우 중요: 
- "그 약", "이 약", "방금 물어본 약", "지금 우리가 얘기하고 있는 약" 등의 표현이 나오면 이전 대화에서 언급된 약물을 의미합니다.
- 사용자가 약물 이름만 묻는다면, 이전 대화에서 언급된 약물의 정확한 이름만 간단히 답변해주세요.
- 사용자가 약물 정보(효능, 용법, 주의사항, 부작용 등)를 묻는다면, 검색된 정보를 바탕으로 상세한 정보를 제공해주세요.
- 질문의 의도를 정확히 파악하여 적절한 수준의 답변을 제공해주세요.

정확성 원칙:
- 제공된 정보에 명확한 내용이 없으면 "죄송하지만 해당 정보를 찾을 수 없습니다"라고 답변하세요.
- 확실하지 않은 정보는 추측하지 말고 "정확한 정보를 확인할 수 없습니다"라고 답변하세요.
- 이전 대화에서 언급되지 않은 약물에 대해 "그 약"이라고 하면 "이전 대화에서 언급된 약물이 없습니다"라고 답변하세요.
- 약물명을 모르거나 찾을 수 없으면 솔직하게 "해당 약물에 대한 정보를 찾을 수 없습니다"라고 답변하세요.

스케줄 생성 가이드:
- 사용자가 약물 복용 스케줄을 요청하거나 "캘린더에 넣어줘", "스케줄 만들어줘" 등의 표현을 사용하면 create_medication_schedule 도구를 사용하세요.
- 스케줄 정보는 "매일 아침 식후 1정", "하루 3번 식후", "저녁 식전 2정" 등의 형태로 파싱하세요.
- 약물명, 단위(tablet/mg/ml/patch/drop), 복용 스케줄 정보를 추출하여 도구에 전달하세요.
- 필수 정보가 부족하면 사용자에게 구체적으로 질문하세요:
  * 약물명이 없으면: "어떤 약물의 복용 스케줄을 만들어드릴까요?"
  * 복용 시간이 없으면: "언제 복용하실 예정인가요? (예: 아침 식후, 하루 3번 등)"
  * 복용량이 없으면: "한 번에 몇 정(또는 mg/ml)씩 복용하실 예정인가요?"

복약 체크리스트 가이드:
- 사용자가 복용 중인 약물 정보를 알려주거나 "복약 정보 추가", "체크리스트에 넣어줘" 등의 표현을 사용하면 add_medication_to_checklist 도구를 사용하세요.
- 복약 정보는 "매끼 식전에 판테놀", "아침 식후에 비타민D", "저녁 식후에 오메가3" 등의 형태로 파싱하세요.
- 약물명, 복용 시간(아침/점심/저녁), 식사 타이밍(식전/식후), 시작일/종료일을 추출하여 도구에 전달하세요.
- 필수 정보가 부족하면 사용자에게 구체적으로 질문하세요.

복약 체크리스트 관리 가이드:
- 사용자가 "내 복약 목록 보여줘", "체크리스트 확인" 등의 요청을 하면 get_user_medications 도구를 사용하세요.
- 사용자가 "복약 정보 수정", "복용 시간 변경" 등의 요청을 하면 update_medication 도구를 사용하세요.
- 사용자가 "복약 정보 삭제", "체크리스트에서 제거" 등의 요청을 하면 delete_medication 도구를 사용하세요.

도구 사용 가이드:
- 약물 정보가 필요하면 search_drug_info 도구를 사용하세요.
- 이전 대화 맥락이 필요하면 get_chat_history 도구를 사용하세요.
- 약물 복용 스케줄 생성이 필요하면 create_medication_schedule 도구를 사용하세요.
- 복약 정보 추가가 필요하면 add_medication_to_checklist 도구를 사용하세요.
- 복약 정보 조회가 필요하면 get_user_medications 도구를 사용하세요.
- 복약 정보 수정이 필요하면 update_medication 도구를 사용하세요.
- 복약 정보 삭제가 필요하면 delete_medication 도구를 사용하세요.
- 필요에 따라 여러 도구를 순차적으로 사용할 수 있습니다."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Agent 생성
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Agent Executor 생성
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor

# 전역 Agent 인스턴스
agent_executor = create_agent()

def initialize_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # 사용자 관련 상태
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    
    if "parsed_medication" not in st.session_state:
        st.session_state.parsed_medication = {}
    
    # 복약 정보 확인 관련 상태
    if "pending_medication_confirmation" not in st.session_state:
        st.session_state.pending_medication_confirmation = None

def add_message(role: str, content: str, sources: List[Dict] = None):
    """메시지 추가"""
    timestamp = datetime.now().strftime("%H:%M")
    message = {
        "role": role,
        "content": content,
        "timestamp": timestamp,
        "sources": sources or []
    }
    st.session_state.messages.append(message)

def parse_medication_info(user_input: str) -> Optional[Dict[str, Any]]:
    """사용자 입력에서 복약 정보를 파싱하는 함수"""
    try:
        # OpenAI LLM을 사용하여 복약 정보 구조화
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        prompt = f"""
다음 사용자 입력에서 복약 정보를 추출하여 JSON 형태로 반환해주세요.
복약 정보가 없다면 null을 반환하세요.

입력: {user_input}

다음 형식으로 반환:
{{
    "medication_name": "약물명",
    "morning": true/false,
    "lunch": true/false,
    "dinner": true/false,
    "before_meal": true/false,
    "after_meal": true/false,
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD 또는 null",
    "is_medication": true/false
}}

한국어로 응답하세요.
"""
        
        response = llm.invoke(prompt).content
        
        # JSON 파싱 시도
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return parsed
            else:
                return None
        except:
            return None
            
    except Exception as e:
        st.write(f"복약 정보 파싱 오류: {e}")
        return None

def classify_user_input(user_input: str) -> Dict:
    """사용자 입력을 분류하는 함수"""
    try:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        prompt = f"""
사용자의 입력을 분석하여 다음 중 하나로 분류해주세요:

입력: {user_input}

분류 기준:
1. "약물 정보 질문": 약물의 효능, 용법, 주의사항, 부작용 등을 묻는 질문
2. "복약 정보": 자신이 복용하고 있는 약물 정보를 알려주는 내용
3. "기타": 위 두 가지에 해당하지 않는 내용

다음 JSON 형태로 응답해주세요:
{{
    "type": "medication_question" | "medication_info" | "other",
    "confidence": 0.0-1.0,
    "reason": "분류 이유",
    "extracted_info": {{
        // medication_info인 경우에만
        "medication_name": "약물명",
        "morning": true/false,
        "lunch": true/false,
        "dinner": true/false,
        "before_meal": true/false,
        "after_meal": true/false,
        "start_date": "YYYY-MM-DD",
        "end_date": "YYYY-MM-DD 또는 null"
    }}
}}

한국어로 응답해주세요.
"""
        
        response = llm.invoke(prompt).content
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                return {"type": "other", "confidence": 0.0, "reason": "JSON 파싱 실패"}
        except:
            return {"type": "other", "confidence": 0.0, "reason": "JSON 파싱 실패"}
            
    except Exception as e:
        return {"type": "other", "confidence": 0.0, "reason": f"분류 오류: {e}"}

def process_medication_info(user_input: str, classification: Dict, med_db: MedicationDatabase, current_user: Dict) -> Dict:
    """복약 정보 처리"""
    try:
        extracted_info = classification.get('extracted_info', {})
        
        # 필수 정보 확인
        if not extracted_info.get('medication_name'):
            return {
                "success": False,
                "message": "약물명을 파악할 수 없습니다. 더 구체적으로 말씀해주세요.",
                "type": "medication_info"
            }
        
        # 시작일이 없으면 오늘 날짜로 설정
        if not extracted_info.get('start_date'):
            extracted_info['start_date'] = datetime.now().date().isoformat()
        
        # 복약 정보 요약 생성
        medication_name = extracted_info['medication_name']
        times = []
        if extracted_info.get('morning'):
            times.append("아침")
        if extracted_info.get('lunch'):
            times.append("점심")
        if extracted_info.get('dinner'):
            times.append("저녁")
        
        meal_timing = []
        if extracted_info.get('before_meal'):
            meal_timing.append("식전")
        if extracted_info.get('after_meal'):
            meal_timing.append("식후")
        
        summary = f"💊 **{medication_name}**\n"
        if times:
            summary += f"📅 **복용 시간:** {', '.join(times)}\n"
        if meal_timing:
            summary += f"🍽️ **식사 타이밍:** {', '.join(meal_timing)}\n"
        if extracted_info.get('start_date'):
            summary += f"📆 **시작일:** {extracted_info['start_date']}\n"
        if extracted_info.get('end_date'):
            summary += f"📆 **종료일:** {extracted_info['end_date']}\n"
        
        # 사용자 확인 요청
        return {
            "success": True,
            "message": f"📋 **복약 정보 확인**\n\n{summary}\n\n위 정보를 복약 체크리스트에 추가하시겠습니까?",
            "type": "medication_info_confirm",
            "medication": extracted_info,
            "needs_confirmation": True
        }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"복약 정보 처리 중 오류가 발생했습니다: {str(e)}",
            "type": "medication_info"
        }

def process_medication_question(user_input: str, conversation_id: str) -> Dict:
    """약물 정보 질문 처리 (Agent 사용)"""
    try:
        result = process_query(user_input, conversation_id)
        return {
            "success": True,
            "message": result["response"],
            "type": "medication_question",
            "sources": result["sources"],
            "chat_history": result["chat_history"],
            "conversation_id": result["conversation_id"]
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"약물 정보 검색 중 오류가 발생했습니다: {str(e)}",
            "type": "medication_question"
        }

def smart_agent_response(user_input: str, med_db: MedicationDatabase, current_user: Dict, conversation_id: str) -> Dict:
    """스마트 에이전트 응답 생성"""
    try:
        # 1단계: 사용자 입력 분류
        classification = classify_user_input(user_input)
        
        # 2단계: 분류에 따른 처리
        if classification['type'] == 'medication_info' and classification['confidence'] > 0.7:
            # 복약 정보 처리
            result = process_medication_info(user_input, classification, med_db, current_user)
            return result
            
        elif classification['type'] == 'medication_question' and classification['confidence'] > 0.7:
            # 약물 정보 질문 처리
            result = process_medication_question(user_input, conversation_id)
            return result
            
        else:
            # 확신도가 낮거나 기타인 경우
            if "약" in user_input or "복용" in user_input or "먹어야" in user_input:
                # 약물 관련으로 추정되는 경우 약물 정보 검색 시도
                result = process_medication_question(user_input, conversation_id)
                return {
                    **result,
                    "message": f"💡 입력을 약물 정보 질문으로 해석했습니다.\n\n{result['message']}\n\n만약 복용 중인 약물 정보를 알려주신 거라면, 더 구체적으로 말씀해주세요. (예: '나 매끼 식전에 판테놀 1정 먹어야 해')"
                }
            else:
                # 일반적인 대화로 처리
                result = process_medication_question(user_input, conversation_id)
                return result
        
    except Exception as e:
        return {
            "success": False,
            "message": f"에이전트 처리 중 오류가 발생했습니다: {str(e)}",
            "type": "error"
        }

def process_query(user_input: str, conversation_id: str = None) -> Dict:
    """Agent를 사용한 쿼리 처리"""
    try:
        # 기존 대화 히스토리 가져오기
        existing_history = st.session_state.get("chat_history", [])
        
        # 대화 히스토리를 LangChain 메시지 형태로 변환
        chat_history = []
        for msg in existing_history[-4:]:  # 최근 4개 대화만 포함
            chat_history.append({"role": "user", "content": msg['user']})
            chat_history.append({"role": "assistant", "content": msg['assistant']})
        
        # Agent 실행
        result = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        
        # 응답에서 AI 메시지 추출
        ai_response = result.get("output", "응답을 생성할 수 없습니다.")
        
        # 새로운 대화 히스토리 업데이트
        new_message = {
            "user": user_input,
            "assistant": ai_response,
            "timestamp": datetime.now().isoformat()
        }
        
        updated_history = existing_history + [new_message]
        if len(updated_history) > 10:
            updated_history = updated_history[-10:]
        
        return {
            "response": ai_response,
            "sources": [],  # Tool 기반에서는 sources 정보가 다르게 처리됨
            "error": "",
            "chat_history": updated_history,
            "conversation_id": conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
    except Exception as e:
        return {
            "response": f"오류가 발생했습니다: {str(e)}",
            "sources": [],
            "error": str(e),
            "chat_history": st.session_state.get("chat_history", []),
            "conversation_id": conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }

def initialize_langchain():
    """LangChain 초기화"""
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        return True, None
    except Exception as e:
        st.error(f"LangChain 초기화 실패: {str(e)}")
        return None, None

def check_qdrant_data():
    """Qdrant 데이터 상태 확인"""
    try:
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        # 컬렉션 정보 확인
        collection_info = qdrant_client.get_collection("product_sections")
        st.write(f"📊 컬렉션 정보: {collection_info}")
        
        # 데이터 개수 확인
        scroll_result = qdrant_client.scroll(
            collection_name="product_sections",
            limit=1,
            with_vectors=True
        )
        st.write(f"📊 총 데이터 개수: {scroll_result[1]}")
        
        # 샘플 데이터 확인
        if scroll_result[0]:
            sample_point = scroll_result[0][0]
            st.write(f"📋 샘플 데이터 payload: {sample_point.payload}")
            st.write(f"📋 샘플 데이터 vector 길이: {len(sample_point.vector) if sample_point.vector else 'None'}")
            
            if not sample_point.vector:
                st.warning("⚠️ 벡터가 없습니다. DB 초기화가 필요합니다.")
            else:
                st.success(f"✅ 벡터가 정상적으로 저장되어 있습니다! (길이: {len(sample_point.vector)})")
        
        # 검색 테스트
        try:
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            search_result = qdrant_client.search(
                collection_name="product_sections",
                query_vector=embeddings.embed_query("타이레놀"),
                limit=1,
                with_payload=True
            )
            
            if search_result:
                st.success(f"🔍 검색 테스트 성공! 검색된 점수: {search_result[0].score:.3f}")
            else:
                st.warning("⚠️ 검색 테스트 실패 - 데이터가 없거나 벡터 문제")
                
        except Exception as e:
            st.error(f"❌ 검색 테스트 실패: {e}")
        
        return True
    except Exception as e:
        st.write(f"❌ Qdrant 데이터 확인 실패: {e}")
        return False

def initialize_qdrant_db():
    """Qdrant 데이터베이스 초기화"""
    try:
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        # 기존 컬렉션 삭제
        try:
            qdrant_client.delete_collection("product_sections")
            st.write("🗑️ 기존 컬렉션 삭제됨")
        except Exception as e:
            st.write(f"📝 기존 컬렉션이 없거나 삭제 실패: {e}")
        
        # 새 컬렉션 생성
        qdrant_client.create_collection(
            collection_name="product_sections",
            vectors_config={"size": 1536, "distance": "Cosine"}
        )
        st.success("✅ 새 컬렉션 생성됨")
        
        st.info("💡 이제 '데이터 재인덱싱' 버튼을 클릭하여 데이터를 업로드하세요!")
        
        return True
    except Exception as e:
        st.error(f"❌ DB 초기화 실패: {e}")
        return False

def reindex_qdrant_data():
    """Qdrant 데이터 재인덱싱"""
    try:
        # JSON 파일에서 데이터 읽기
        json_path = "../all_drug_data.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        st.write(f"📊 JSON 데이터 로드됨: {len(data)}개 항목")
        
        # Qdrant 클라이언트
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        # Embeddings 초기화
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 데이터 구조 확인 및 처리
        st.write(f"📊 데이터 타입: {type(data)}")
        if isinstance(data, dict):
            st.write(f"📊 딕셔너리 키: {list(data.keys())}")
            
            # 모든 약물 데이터 수집
            all_items = []
            for drug_name, drug_list in data.items():
                if isinstance(drug_list, list):
                    for drug_item in drug_list:
                        if isinstance(drug_item, dict):
                            drug_item['aliases'] = drug_item.get('aliases', []) + [drug_name]
                        all_items.append(drug_item)
            
            items_to_process = all_items
            st.write(f"📊 처리할 약물 수: {len(items_to_process)}")
        else:
            st.error("❌ 데이터가 딕셔너리 형태가 아닙니다!")
            return False
        
        # 데이터 업로드
        uploaded_count = 0
        for i, item in enumerate(items_to_process):
            try:
                # 필수 필드 확인
                if not item.get('efcyQesitm') or not item.get('efcyQesitm').strip():
                    st.write(f"⚠️ {i+1}번째 항목: 빈 텍스트로 건너뜀")
                    continue
                
                # 섹션별로 데이터 분리
                sections = {
                    'efficacy': item.get('efcyQesitm', '').strip(),
                    'usage': item.get('useMethodQesitm', '').strip(),
                    'warning': item.get('atpnWarnQesitm', '').strip(),
                    'precaution': item.get('atpnQesitm', '').strip(),
                    'interaction': item.get('intrcQesitm', '').strip(),
                    'side_effect': item.get('seQesitm', '').strip(),
                    'storage': item.get('depositMethodQesitm', '').strip()
                }
                
                # 각 섹션을 개별 문서로 업로드
                for section_name, section_text in sections.items():
                    if section_text and section_text.strip():
                        # 벡터 생성
                        vector = embeddings.embed_query(section_text)
                        
                        # Qdrant에 업로드
                        qdrant_client.upsert(
                            collection_name="product_sections",
                            points=[{
                                "id": f"{item.get('itemSeq', '')}_{section_name}_{uploaded_count}",
                                "vector": vector,
                                "payload": {
                                    "text": section_text,
                                    "item_name": item.get('itemName', ''),
                                    "entp_name": item.get('entpName', ''),
                                    "section": section_name,
                                    "section_name": section_name,
                                    "aliases": item.get('aliases', []),
                                    "ingredients": item.get('ingredients', []),
                                    "update_de": item.get('updateDe', ''),
                                    "is_otc": item.get('is_otc', False)
                                }
                            }]
                        )
                        uploaded_count += 1
                
                st.write(f"✅ {i+1}번째 항목 처리됨: {item.get('itemName', '')}... (섹션별로 분리)")
                
            except Exception as e:
                st.write(f"❌ {i+1}번째 항목 처리 실패: {e}")
                st.write(f"📋 실패한 항목 데이터: {item}")
        
        st.success(f"✅ {uploaded_count}개 섹션 업로드 완료!")
        return True
        
    except Exception as e:
        st.error(f"❌ 재인덱싱 실패: {e}")
        return False

def main():
    """메인 앱"""
    initialize_session_state()
    
    # 데이터베이스 초기화
    med_db = MedicationDatabase()
    
    # 헤더
    st.markdown('<h1 class="main-header">💊 Medication Agent</h1>', unsafe_allow_html=True)
    st.markdown("### 🤖 Tool 기반 AI 약물 정보 챗봇 + 복약 체크리스트")
    
    # 사이드바
    with st.sidebar:
        st.header("🔧 설정")
        
        # 사용자 로그인/회원가입
        st.subheader("👤 사용자 관리")
        
        if not st.session_state.current_user:
            # 통합된 로그인/회원가입
            username = st.text_input("사용자명", placeholder="사용자명을 입력하세요", key="username_input")
            email = st.text_input("이메일 (선택사항)", placeholder="이메일을 입력하세요", key="email_input")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔐 로그인/회원가입", type="primary", key="login_signup_button"):
                    if username:
                        user = med_db.get_or_create_user(username, email)
                        if user:
                            st.session_state.current_user = user
                            st.success(f"✅ {username}으로 로그인되었습니다!")
                            st.rerun()
                    else:
                        st.error("사용자명을 입력해주세요.")
        else:
            st.success(f"✅ {st.session_state.current_user['name']}으로 로그인됨")
            
            if st.button("🚪 로그아웃"):
                st.session_state.current_user = None
                st.rerun()
        
        st.markdown("---")
        
        # 복약 정보 예시
        st.subheader("💊 복약 정보 예시")
        medication_examples = [
            "나 매끼 식전에 판테놀 먹어야 해",
            "아침 식후에 비타민D 복용",
            "저녁 식후에 오메가3 먹어야 함"
        ]
        
        for i, example in enumerate(medication_examples):
            if st.button(example, key=f"sidebar_medication_{i}"):
                st.session_state.example_medication = example
                st.rerun()
        
        st.markdown("---")
        
        # 복약 체크리스트 관리
        if st.session_state.current_user:
            st.subheader("💊 복약 체크리스트")
            
            user_medications = med_db.get_user_medications(st.session_state.current_user['id'])
            
            if user_medications:
                st.write(f"📋 총 {len(user_medications)}개 복약")
                
                for med in user_medications:
                    with st.expander(f"💊 {med['medication_name']}"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            times = []
                            if med['morning']:
                                times.append("아침")
                            if med['lunch']:
                                times.append("점심")
                            if med['dinner']:
                                times.append("저녁")
                            st.write(f"**복용 시간:** {', '.join(times)}")
                            
                            meal_timing = []
                            if med['before_meal']:
                                meal_timing.append("식전")
                            if med['after_meal']:
                                meal_timing.append("식후")
                            if meal_timing:
                                st.write(f"**식사 타이밍:** {', '.join(meal_timing)}")
                            
                            if med['start_date']:
                                st.write(f"**시작일:** {med['start_date']}")
                            if med['end_date']:
                                st.write(f"**종료일:** {med['end_date']}")
                        
                        with col2:
                            if st.button("✏️ 수정", key=f"edit_{med['id']}"):
                                st.session_state.editing_medication = med
                                st.rerun()
                            
                            if st.button("🗑️ 삭제", key=f"delete_{med['id']}"):
                                if med_db.delete_medication(med['id']):
                                    st.rerun()
            else:
                st.info("💊 복약 정보를 입력하면 자동으로 체크리스트에 추가됩니다!")
        
        st.markdown("---")
        
        # 검색 옵션
        st.subheader("검색 옵션")
        
        # 결과 수 설정
        k = st.slider("📊 검색 결과 수", 1, 10, 3)
        
        st.markdown("---")
        
        # 대화 초기화
        if st.button("🗑️ 대화 초기화", type="secondary"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.rerun()
        
        st.markdown("---")
        
        # 시스템 상태
        st.subheader("📊 시스템 상태")
        
        # LangChain 초기화 상태 확인
        try:
            qa_chain, vectorstore = initialize_langchain()
            if qa_chain:
                st.success("✅ Tool Agent 정상")
            else:
                st.error("❌ Tool Agent 초기화 실패")
        except Exception as e:
            st.error(f"❌ 연결 오류: {str(e)}")
        
        # Qdrant 데이터 확인 버튼
        if st.button("🔍 Qdrant 데이터 확인", type="secondary"):
            check_qdrant_data()
        
        # DB 초기화 버튼
        if st.button("🗄️ DB 초기화", type="secondary"):
            initialize_qdrant_db()
        
        # 재인덱싱 버튼
        if st.button("🔄 데이터 재인덱싱", type="secondary"):
            reindex_qdrant_data()
        
        st.markdown("---")
        st.subheader("📅 스케줄 파일")
        
        # 스케줄 파일 목록 표시
        if os.path.exists(EXPORT_DIR):
            files = [f for f in os.listdir(EXPORT_DIR) if f.endswith(('.json', '.ics'))]
            if files:
                st.write(f"📁 생성된 파일: {len(files)}개")
                with st.expander("📋 파일 목록"):
                    for file in sorted(files, reverse=True)[:10]:  # 최근 10개만
                        file_path = os.path.join(EXPORT_DIR, file)
                        file_size = os.path.getsize(file_path)
                        st.write(f"📄 {file} ({file_size} bytes)")
            else:
                st.info("📁 아직 생성된 스케줄 파일이 없습니다.")
        else:
            st.info("📁 스케줄 파일 폴더가 아직 생성되지 않았습니다.")
        
        # 폴더 열기 버튼
        if st.button("📂 스케줄 폴더 열기", type="secondary"):
            if os.path.exists(EXPORT_DIR):
                st.success(f"📂 폴더 경로: {os.path.abspath(EXPORT_DIR)}")
            else:
                st.info("📁 아직 스케줄 파일이 생성되지 않았습니다.")
    
    # 메인 채팅 영역
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 복약 정보 수정 모달
        if 'editing_medication' in st.session_state:
            med = st.session_state.editing_medication
            
            with st.container():
                st.markdown("### ✏️ 복약 정보 수정")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    medication_name = st.text_input("약물명", value=med['medication_name'])
                    start_date = st.date_input("시작일", value=datetime.strptime(med['start_date'], '%Y-%m-%d').date())
                    end_date = st.date_input("종료일", value=datetime.strptime(med['end_date'], '%Y-%m-%d').date() if med['end_date'] else None)
                
                with col_b:
                    st.write("**복용 시간**")
                    morning = st.checkbox("아침", value=med['morning'])
                    lunch = st.checkbox("점심", value=med['lunch'])
                    dinner = st.checkbox("저녁", value=med['dinner'])
                    
                    st.write("**식사 타이밍**")
                    before_meal = st.checkbox("식전", value=med['before_meal'])
                    after_meal = st.checkbox("식후", value=med['after_meal'])
                    
                    if st.button("✅ 수정 완료"):
                        update_data = {
                            'medication_name': medication_name,
                            'morning': morning,
                            'lunch': lunch,
                            'dinner': dinner,
                            'before_meal': before_meal,
                            'after_meal': after_meal,
                            'start_date': start_date.isoformat(),
                            'end_date': end_date.isoformat() if end_date else None
                        }
                        
                        if med_db.update_medication(med['id'], update_data):
                            del st.session_state.editing_medication
                            st.rerun()
                    
                    if st.button("❌ 취소"):
                        del st.session_state.editing_medication
                        st.rerun()
        
        # 복약 정보 예시가 있으면 자동으로 전송
        if 'example_medication' in st.session_state and st.session_state.example_medication:
            user_input = st.session_state.example_medication
            if user_input.strip() and st.session_state.current_user:
                add_message("user", user_input)
                
                with st.spinner("🤖 복약 정보 분석 중..."):
                    result = smart_agent_response(user_input, med_db, st.session_state.current_user, st.session_state.conversation_id)
                    
                    if result["success"]:
                        ai_response = result["message"]
                        sources = []
                    else:
                        ai_response = result["message"]
                        sources = []
                
                add_message("assistant", ai_response, sources)
                del st.session_state.example_medication
                st.rerun()
        # 채팅 메시지 표시
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # 복약 정보 확인 메시지인 경우 확인 버튼 표시
                if (message["role"] == "assistant" and 
                    "복약 정보 확인" in message["content"] and
                    st.session_state.pending_medication_confirmation):
                    
                    st.markdown("---")
                    col_confirm1, col_confirm2 = st.columns(2)
                    
                    with col_confirm1:
                        if st.button("✅ 확인 - 체크리스트에 추가", key=f"confirm_med_{i}", type="primary"):
                            # 복약 정보를 DB에 추가
                            medication_data = st.session_state.pending_medication_confirmation
                            if med_db.add_medication(st.session_state.current_user['id'], medication_data):
                                # 메시지 업데이트
                                st.session_state.messages[i]["content"] = f"✅ **복약 정보가 체크리스트에 추가되었습니다!**\n\n{message['content'].replace('위 정보를 복약 체크리스트에 추가하시겠습니까?', '')}"
                                st.session_state.pending_medication_confirmation = None
                                st.rerun()
                    
                    with col_confirm2:
                        if st.button("❌ 취소", key=f"cancel_med_{i}"):
                            # 메시지 업데이트
                            st.session_state.messages[i]["content"] = f"❌ **복약 정보 추가가 취소되었습니다.**\n\n{message['content'].replace('위 정보를 복약 체크리스트에 추가하시겠습니까?', '')}"
                            st.session_state.pending_medication_confirmation = None
                            st.rerun()
        
        # 입력 영역
        st.markdown("---")
        
        # 질문 입력 필드
        user_input = st.text_input(
            "💬 질문을 입력하세요...",
            placeholder="예: 타이레놀의 효능이 뭔가요? 또는: 나 매끼 식전에 판테놀 먹어야 해",
            key="user_input"
        )
        
        # 복약 정보 자동 감지 및 체크리스트 추가
        if user_input.strip() and st.session_state.current_user:
            parsed_med = parse_medication_info(user_input)
            if parsed_med and parsed_med.get('is_medication'):
                st.session_state.parsed_medication = parsed_med
                
                with st.expander("💊 복약 정보 감지됨", expanded=True):
                    st.write(f"**약물명:** {parsed_med['medication_name']}")
                    
                    times = []
                    if parsed_med.get('morning'):
                        times.append("아침")
                    if parsed_med.get('lunch'):
                        times.append("점심")
                    if parsed_med.get('dinner'):
                        times.append("저녁")
                    if times:
                        st.write(f"**복용 시간:** {', '.join(times)}")
                    
                    meal_timing = []
                    if parsed_med.get('before_meal'):
                        meal_timing.append("식전")
                    if parsed_med.get('after_meal'):
                        meal_timing.append("식후")
                    if meal_timing:
                        st.write(f"**식사 타이밍:** {', '.join(meal_timing)}")
                    
                    if parsed_med.get('start_date'):
                        st.write(f"**시작일:** {parsed_med['start_date']}")
                    if parsed_med.get('end_date'):
                        st.write(f"**종료일:** {parsed_med['end_date']}")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("✅ 체크리스트에 추가", key="add_to_checklist", type="primary"):
                            if med_db.add_medication(st.session_state.current_user['id'], parsed_med):
                                st.session_state.parsed_medication = {}
                                st.rerun()
                    
                    with col_b:
                        if st.button("❌ 추가하지 않음", key="dont_add"):
                            st.session_state.parsed_medication = {}
                            st.rerun()
        
        # 전송 버튼
        col_a, col_b, col_c = st.columns([1, 1, 1])
        
        with col_a:
            if st.button("🚀 전송", type="primary", use_container_width=True):
                if user_input.strip():
                    # 사용자 메시지 추가
                    add_message("user", user_input)
                    
                    # 스마트 에이전트 실행
                    with st.spinner("🤖 스마트 에이전트 분석 중..."):
                        if st.session_state.current_user:
                            result = smart_agent_response(user_input, med_db, st.session_state.current_user, st.session_state.conversation_id)
                        else:
                            result = process_medication_question(user_input, st.session_state.conversation_id)
                        
                        if result["success"]:
                            ai_response = result["message"]
                            sources = result.get("sources", [])
                            
                            # 복약 정보 확인이 필요한 경우 세션 상태 설정
                            if result["type"] == "medication_info_confirm" and result.get("needs_confirmation"):
                                st.session_state.pending_medication_confirmation = result.get("medication")
                            
                            if result["type"] == "medication_question":
                                st.session_state.chat_history = result.get("chat_history", [])
                                st.session_state.conversation_id = result.get("conversation_id", st.session_state.conversation_id)
                            else:
                                ai_response = result["message"]
                                sources = []
                        else:
                            ai_response = result["message"]
                            sources = []
                    
                    # AI 메시지 추가
                    add_message("assistant", ai_response, sources)
                    
                    # 페이지 새로고침
                    st.rerun()
                else:
                    st.error("질문을 입력해주세요.")
        
        with col_c:
            st.write("")  # 빈 공간
    
    with col2:
        st.markdown("### 📈 대화 통계")
        st.metric("총 메시지", len(st.session_state.messages))
        st.metric("사용자 메시지", len([m for m in st.session_state.messages if m["role"] == "user"]))
        st.metric("AI 응답", len([m for m in st.session_state.messages if m["role"] == "assistant"]))
        
        st.markdown("---")
        
        if st.session_state.current_user:
            st.markdown("### 💊 복약 현황")
            
            user_medications = med_db.get_user_medications(st.session_state.current_user['id'])
            
            if user_medications:
                total = len(user_medications)
                
                today = datetime.now().date()
                today_meds = 0
                
                for med in user_medications:
                    # start_date가 문자열이면 datetime 객체로 변환
                    start_date = med['start_date']
                    if isinstance(start_date, str):
                        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
                    
                    if start_date <= today:
                        if med['morning'] or med['lunch'] or med['dinner']:
                            today_meds += 1
                
                st.metric("총 복약", total)
                st.metric("오늘 복용", today_meds)
            else:
                st.info("복약 체크리스트가 비어있습니다.")
        
        st.markdown("---")
        
        st.markdown("### 💬 채팅 히스토리")
        st.metric("대화 세션", st.session_state.conversation_id)
        st.metric("히스토리 길이", len(st.session_state.chat_history))
        
        # 최근 대화 미리보기
        if st.session_state.chat_history:
            with st.expander("📝 최근 대화 미리보기"):
                for i, msg in enumerate(st.session_state.chat_history[-3:], 1):
                    st.markdown(f"**{i}. 사용자:** {msg['user'][:50]}...")
                    st.markdown(f"**AI:** {msg['assistant'][:50]}...")
                    st.markdown("---")

if __name__ == "__main__":
    main()
