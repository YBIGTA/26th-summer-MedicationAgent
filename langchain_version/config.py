"""
💊 Medication Agent - 설정 및 상수
애플리케이션 전반에서 사용되는 설정값과 상수들을 관리합니다.
"""

import os
from datetime import date, time
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# ========== 기본 설정 ==========

# 타임존 및 지역 설정
TIMEZONE = os.getenv("APP_TZ", "Asia/Seoul")
EXPORT_DIR = os.getenv("CAL_EXPORT_DIR", "./_calendar_exports")

# 기본 식사 시각
DEFAULT_MEAL_TIMES = {
    "morning": time(8, 0),   # 아침 08:00
    "lunch":   time(12, 30), # 점심 12:30
    "evening": time(19, 0),  # 저녁 19:00
}

# 복용 시간 오프셋 (분)
PRE_MEAL_OFFSET_MIN = 30   # 식전 30분
POST_MEAL_OFFSET_MIN = 30  # 식후 30분

# 주간 시작 날짜 (2025-08-24 일요일부터)
WEEK_START_DATE = date(2025, 8, 24)

# 한국 요일명 (datetime.weekday() 대응)
WEEK_DAYS_KO = ["월", "화", "수", "목", "금", "토", "일"]

# ========== LangChain 설정 ==========

# OpenAI 모델 설정
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "4000"))

# Qdrant 설정
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "medication_data")

# ========== 데이터베이스 설정 ==========

# Supabase PostgreSQL 설정
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

# ========== UI 설정 ==========

# 페이지 설정
PAGE_TITLE = "💊 Medication Agent - AI 기반 약물 정보 챗봇"
PAGE_ICON = "💊"
LAYOUT = "wide"

# 세션 상태 키들
SESSION_KEYS = {
    "messages": "messages",
    "conversation_id": "conversation_id",
    "current_user": "current_user",
    "med_db": "med_db",
    "agent": "agent",
    "qdrant_client": "qdrant_client",
    "qa_chain": "qa_chain",
    "test_mode": "test_mode"
}

# ========== 약물 정보 설정 ==========

# 약물 데이터 파일 경로
DRUG_DATA_FILE = "all_drug_data.json"

# 검색 결과 수 제한
MAX_SEARCH_RESULTS = 5

# ========== 캘린더 설정 ==========

# 캘린더 이벤트 설정
CALENDAR_EVENT_DURATION_MINUTES = 10
CALENDAR_EXPORT_FORMATS = ["ics", "json"]

# ========== 프롬프트 템플릿 ==========

# 약물 정보 검색 프롬프트
DRUG_SEARCH_PROMPT = """
당신은 한국의 약물 정보 전문가입니다. 
사용자의 질문에 대해 정확하고 이해하기 쉽게 답변해주세요.

참고 정보: {context}

질문: {question}

답변은 다음 형식을 따라주세요:
1. 간단한 요약
2. 상세한 설명
3. 주의사항 (있는 경우)
4. 추가 정보 (있는 경우)

한국어로 답변해주세요.
"""

# 복약 일정 생성 프롬프트
MEDICATION_SCHEDULE_PROMPT = """
사용자가 복용하려는 약물의 일정을 생성해주세요.
다음 정보를 바탕으로 7일간의 복용 일정을 만들어주세요:

약물명: {drug_name}
단위: {unit}
복용 정보: {schedule_info}

응답은 다음 JSON 형식으로 제공해주세요:
[[CALENDAR_JSON]]
{{
  "drug_name": "약물명",
  "unit": "단위",
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

# ========== 에러 메시지 ==========

ERROR_MESSAGES = {
    "qdrant_connection": "❌ Qdrant 연결에 실패했습니다. 환경변수를 확인해주세요.",
    "openai_connection": "❌ OpenAI 연결에 실패했습니다. API 키를 확인해주세요.",
    "database_connection": "❌ 데이터베이스 연결에 실패했습니다.",
    "no_drug_data": "❌ 약물 데이터를 찾을 수 없습니다.",
    "invalid_input": "❌ 잘못된 입력입니다. 다시 시도해주세요."
}

# ========== 성공 메시지 ==========

SUCCESS_MESSAGES = {
    "qdrant_initialized": "✅ Qdrant 데이터베이스가 초기화되었습니다.",
    "data_indexed": "✅ 약물 데이터가 인덱싱되었습니다.",
    "medication_added": "✅ 약물이 체크리스트에 추가되었습니다.",
    "medication_updated": "✅ 약물 정보가 업데이트되었습니다.",
    "medication_deleted": "✅ 약물이 삭제되었습니다.",
    "calendar_exported": "✅ 캘린더가 내보내졌습니다."
}
