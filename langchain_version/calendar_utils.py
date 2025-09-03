"""
💊 Medication Agent - 캘린더 유틸리티
복약 일정을 캘린더 형식으로 변환하고 내보내는 기능을 제공합니다.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

from config import (
    TIMEZONE, EXPORT_DIR, DEFAULT_MEAL_TIMES, 
    PRE_MEAL_OFFSET_MIN, POST_MEAL_OFFSET_MIN, 
    WEEK_DAYS_KO, CALENDAR_EVENT_DURATION_MINUTES
)


def ensure_dir(path: str) -> None:
    """디렉토리가 존재하지 않으면 생성합니다."""
    os.makedirs(path, exist_ok=True)


def parse_calendar_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    AI 응답 안의 [[CALENDAR_JSON]] ... [[/CALENDAR_JSON]] 블록을 찾아 JSON 파싱.
    
    Args:
        text (str): AI 응답 텍스트
        
    Returns:
        Optional[Dict[str, Any]]: 파싱된 JSON 데이터 또는 None
    """
    pattern = r"\[\[CALENDAR_JSON\]\]\s*(\{.*?\})\s*\[\[/CALENDAR_JSON\]\]"
    match = re.search(pattern, text, re.DOTALL)
    
    if not match:
        return None
    
    try:
        return json.loads(match.group(1))
    except Exception:
        return None


def slot_to_datetime(d: date, slot_key: str, tz: str) -> datetime:
    """
    슬롯 키를 기반으로 날짜와 시간을 결합합니다.
    
    Args:
        d (date): 날짜
        slot_key (str): 슬롯 키 (morning_before, morning_after 등)
        tz (str): 타임존
        
    Returns:
        datetime: 계산된 날짜시간
    """
    meal, when = slot_key.split("_")
    base_time = DEFAULT_MEAL_TIMES[meal]
    base_dt = datetime.combine(d, base_time, tzinfo=ZoneInfo(tz))
    
    if when == "before":
        return base_dt - timedelta(minutes=PRE_MEAL_OFFSET_MIN)
    else:
        return base_dt + timedelta(minutes=POST_MEAL_OFFSET_MIN)


def build_week_events(plan: Dict[str, Any], start_date: date, tz: str) -> List[Dict[str, Any]]:
    """
    복약 계획을 기반으로 주간 이벤트 리스트를 생성합니다.
    
    Args:
        plan (Dict[str, Any]): 복약 계획
        start_date (date): 시작 날짜
        tz (str): 타임존
        
    Returns:
        List[Dict[str, Any]]: 이벤트 리스트
    """
    events = []
    per_slot: Dict[str, Any] = plan.get("per_slot", {})
    drug = plan.get("drug_name", "약물")
    unit = plan.get("unit", "tablet")
    notes = plan.get("notes", "")
    
    # 시간대별 라벨 매핑
    label_map = {
        "morning": "아침", "lunch": "점심", "evening": "저녁",
        "before": "식전", "after": "식후"
    }
    
    for day_offset in range(7):
        current_date = start_date + timedelta(days=day_offset)
        weekday_ko = WEEK_DAYS_KO[current_date.weekday()]
        
        for slot_key in [
            "morning_before", "morning_after", 
            "lunch_before", "lunch_after", 
            "evening_before", "evening_after"
        ]:
            val = per_slot.get(slot_key, 0)
            
            try:
                count = int(val)
            except Exception:
                continue
                
            if count <= 0:
                continue
            
            start_dt = slot_to_datetime(current_date, slot_key, tz)
            end_dt = start_dt + timedelta(minutes=CALENDAR_EVENT_DURATION_MINUTES)
            
            meal, when = slot_key.split("_")
            human_slot = f"{label_map[meal]} {label_map[when]}"
            
            events.append({
                "summary": f"{drug} {count}{unit}",
                "description": f"{human_slot} 복용\n{notes}",
                "start": start_dt,
                "end": end_dt,
                "date": current_date,
                "weekday": weekday_ko,
                "slot": slot_key,
                "count": count,
                "unit": unit
            })
    
    return events


def export_json(events: List[Dict[str, Any]], out_path: str) -> None:
    """
    이벤트를 JSON 형식으로 내보냅니다.
    
    Args:
        events (List[Dict[str, Any]]): 이벤트 리스트
        out_path (str): 출력 파일 경로
    """
    ensure_dir(os.path.dirname(out_path))
    
    # datetime 객체를 문자열로 변환
    exportable_events = []
    for event in events:
        exportable_event = event.copy()
        exportable_event["start"] = event["start"].isoformat()
        exportable_event["end"] = event["end"].isoformat()
        exportable_event["date"] = event["date"].isoformat()
        exportable_events.append(exportable_event)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(exportable_events, f, ensure_ascii=False, indent=2)


def escape_ics(text: str) -> str:
    """ICS 파일에서 사용할 수 있도록 텍스트를 이스케이프합니다."""
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def dt_local(dt: datetime) -> str:
    """datetime을 로컬 시간대 문자열로 변환합니다."""
    return dt.strftime("%Y%m%dT%H%M%S")


def export_ics(events: List[Dict[str, Any]], out_path: str, tz: str) -> None:
    """
    이벤트를 ICS (iCalendar) 형식으로 내보냅니다.
    
    Args:
        events (List[Dict[str, Any]]): 이벤트 리스트
        out_path (str): 출력 파일 경로
        tz (str): 타임존
    """
    ensure_dir(os.path.dirname(out_path))
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("BEGIN:VCALENDAR\n")
        f.write("VERSION:2.0\n")
        f.write(f"PRODID:-//Medication Agent//KR\n")
        f.write(f"TZID:{tz}\n")
        f.write("BEGIN:VTIMEZONE\n")
        f.write(f"TZID:{tz}\n")
        f.write("END:VTIMEZONE\n")
        
        for event in events:
            f.write("BEGIN:VEVENT\n")
            f.write(f"UID:{event['start'].strftime('%Y%m%d%H%M%S')}@medication-agent.com\n")
            f.write(f"DTSTART:{dt_local(event['start'])}\n")
            f.write(f"DTEND:{dt_local(event['end'])}\n")
            f.write(f"SUMMARY:{escape_ics(event['summary'])}\n")
            f.write(f"DESCRIPTION:{escape_ics(event['description'])}\n")
            f.write("END:VEVENT\n")
        
        f.write("END:VCALENDAR\n")


def export_calendar(events: List[Dict[str, Any]], format_type: str = "ics", 
                   filename: str = None, tz: str = None) -> str:
    """
    이벤트를 지정된 형식으로 내보냅니다.
    
    Args:
        events (List[Dict[str, Any]]): 이벤트 리스트
        format_type (str): 내보낼 형식 ("ics" 또는 "json")
        filename (str): 파일명 (없으면 자동 생성)
        tz (str): 타임존 (없으면 기본값 사용)
        
    Returns:
        str: 내보낸 파일 경로
    """
    if not events:
        return ""
    
    if tz is None:
        tz = TIMEZONE
    
    if filename is None:
        drug_name = events[0].get("summary", "medication").split()[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{drug_name}_{timestamp}"
    
    ensure_dir(EXPORT_DIR)
    
    if format_type.lower() == "ics":
        out_path = os.path.join(EXPORT_DIR, f"{filename}.ics")
        export_ics(events, out_path, tz)
    elif format_type.lower() == "json":
        out_path = os.path.join(EXPORT_DIR, f"{filename}.json")
        export_json(events, out_path)
    else:
        raise ValueError(f"지원하지 않는 형식: {format_type}")
    
    return out_path


def get_week_start_date() -> date:
    """현재 주의 시작 날짜를 반환합니다."""
    today = date.today()
    days_since_monday = today.weekday()
    return today - timedelta(days=days_since_monday)


def format_time_range(start_time: time, end_time: time) -> str:
    """시간 범위를 읽기 쉬운 형식으로 변환합니다."""
    return f"{start_time.strftime('%H:%M')} ~ {end_time.strftime('%H:%M')}"


def get_meal_time_info() -> Dict[str, Dict[str, Any]]:
    """식사 시간 정보를 반환합니다."""
    meal_info = {}
    
    for meal, time_obj in DEFAULT_MEAL_TIMES.items():
        meal_info[meal] = {
            "time": time_obj,
            "before": time_obj - timedelta(minutes=PRE_MEAL_OFFSET_MIN),
            "after": time_obj + timedelta(minutes=POST_MEAL_OFFSET_MIN),
            "before_str": (time_obj - timedelta(minutes=PRE_MEAL_OFFSET_MIN)).strftime("%H:%M"),
            "after_str": (time_obj + timedelta(minutes=POST_MEAL_OFFSET_MIN)).strftime("%H:%M")
        }
    
    return meal_info
