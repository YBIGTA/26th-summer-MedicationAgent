import json
import os
from typing import Dict, List

from langchain.tools import tool


DATA_FILE = os.path.join(os.path.dirname(__file__), "data.json")


def _load_medication_data() -> List[Dict]:
	"""Load medication data from data.json."""
	if not os.path.exists(DATA_FILE):
		raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {DATA_FILE}")
	with open(DATA_FILE, "r", encoding="utf-8") as f:
		return json.load(f)


def _find_medication_by_name(name: str) -> Dict:
	"""Find a medication entry by name (case-insensitive, exact or partial match)."""
	name_norm = name.strip().lower()
	data = _load_medication_data()

	# Try exact match first
	for entry in data:
		if entry.get("name", "").strip().lower() == name_norm:
			return entry

	# Fallback to partial match
	for entry in data:
		if name_norm in entry.get("name", "").strip().lower():
			return entry

	return {}


@tool("lookup_medication", return_direct=False)
def lookup_medication(name: str) -> str:
	"""
	의약품 정보를 조회한다. 입력: 약 이름(예: '이부프로펜', '아세트아미노펜').
	가능하면 정확한 제품/성분명을 전달하라. 응답에는 용도, 용법, 주의사항, 부작용,
	금기 및 상호작용이 포함될 수 있다. 찾지 못한 경우 이유를 반환한다.
	"""
	if not name or not name.strip():
		return "약 이름이 비어 있습니다. 예: '이부프로펜'"

	entry = _find_medication_by_name(name)
	if not entry:
		return f"'{name}'에 대한 정보를 찾지 못했습니다. 철자 또는 다른 이름을 시도하세요."

	parts: List[str] = []
	parts.append(f"약 이름: {entry.get('name', 'N/A')}")
	if entry.get("uses"):
		parts.append(f"용도: {entry['uses']}")
	if entry.get("dosage"):
		parts.append(f"일반 용법/용량: {entry['dosage']}")
	if entry.get("precautions"):
		parts.append(f"주의사항: {entry['precautions']}")
	if entry.get("side_effects"):
		parts.append(f"주요 부작용: {entry['side_effects']}")
	if entry.get("contraindications"):
		parts.append(f"금기: {entry['contraindications']}")
	if entry.get("interactions"):
		parts.append(f"상호작용: {entry['interactions']}")

	return "\n".join(parts)


def get_tools():
	"""Return the list of tools available to the agent."""
	return [lookup_medication]


__all__ = ["lookup_medication", "get_tools"]

