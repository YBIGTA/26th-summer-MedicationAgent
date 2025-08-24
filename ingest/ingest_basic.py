#!/usr/bin/env python3
"""
기본 약물 데이터 인제스트 스크립트 (PostgreSQL만)
API 키 없이도 기본 데이터 구조를 테스트할 수 있습니다.
"""

import json
import os
import re
from typing import Dict, List, Any
import psycopg2
from psycopg2.extras import RealDictCursor

# 설정
DATA_FILE = "../all_drug_data.json"

# 섹션 매핑
SECTION_MAPPING = {
    "efcyQesitm": "efficacy",
    "useMethodQesitm": "dosage", 
    "atpnWarnQesitm": "warnings",
    "atpnQesitm": "precautions",
    "intrcQesitm": "interactions",
    "seQesitm": "side_effects",
    "depositMethodQesitm": "storage"
}

def get_db_connection():
    """PostgreSQL 연결을 반환합니다."""
    return psycopg2.connect(
        host="localhost",
        database="medication_agent",
        user="parkhaneul",
        sslmode="disable"
    )

def extract_ingredients(item_name: str) -> List[str]:
    """제품명에서 성분을 추출합니다."""
    ingredients = []
    # 괄호 안의 성분 추출
    matches = re.findall(r'\(([^)]+)\)', item_name)
    for match in matches:
        # 쉼표나 공백으로 분리
        parts = re.split(r'[,·\s]+', match)
        ingredients.extend([p.strip() for p in parts if p.strip()])
    return ingredients

def split_text(text: str, max_length: int = 1000) -> List[str]:
    """텍스트를 최대 길이로 분할합니다."""
    if len(text) <= max_length:
        return [text]
    
    parts = []
    current_part = ""
    
    for sentence in re.split(r'[.!?]', text):
        if len(current_part) + len(sentence) <= max_length:
            current_part += sentence + "."
        else:
            if current_part:
                parts.append(current_part.strip())
            current_part = sentence + "."
    
    if current_part:
        parts.append(current_part.strip())
    
    return parts

def upsert_product(conn, item_seq: str, data: Dict[str, Any]):
    """제품 정보를 upsert합니다."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO products (item_seq, entp_name, item_name, item_image, bizrno, open_de, update_de, raw_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (item_seq) DO UPDATE SET
                entp_name = EXCLUDED.entp_name,
                item_name = EXCLUDED.item_name,
                item_image = EXCLUDED.item_image,
                bizrno = EXCLUDED.bizrno,
                open_de = EXCLUDED.open_de,
                update_de = EXCLUDED.update_de,
                raw_json = EXCLUDED.raw_json
        """, (
            item_seq,
            data.get("entpName"),
            data.get("itemName"),
            data.get("itemImage"),
            data.get("bizrno"),
            data.get("openDe")[:10] if data.get("openDe") else None,
            data.get("updateDe")[:10] if data.get("updateDe") else None,
            json.dumps(data, ensure_ascii=False)
        ))

def insert_product_alias(conn, alias: str, item_seq: str):
    """제품 별칭을 삽입합니다."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO product_aliases (alias, item_seq)
            VALUES (%s, %s)
            ON CONFLICT (alias, item_seq) DO NOTHING
        """, (alias, item_seq))

def insert_product_ingredients(conn, item_seq: str, ingredients: List[str]):
    """제품 성분을 삽입합니다."""
    with conn.cursor() as cur:
        for ingredient in ingredients:
            cur.execute("""
                INSERT INTO product_ingredients (item_seq, ingredient)
                VALUES (%s, %s)
                ON CONFLICT (item_seq, ingredient) DO NOTHING
            """, (item_seq, ingredient))

def upsert_product_sections(conn, item_seq: str, section: str, text_parts: List[str]):
    """제품 섹션을 upsert합니다."""
    with conn.cursor() as cur:
        for part_idx, text in enumerate(text_parts):
            cur.execute("""
                INSERT INTO product_sections (item_seq, section, part_idx, text)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (item_seq, section, part_idx) DO UPDATE SET
                    text = EXCLUDED.text
            """, (item_seq, section, part_idx, text))

def main():
    """메인 함수"""
    print("기본 약물 데이터 인제스트 시작...")
    
    # 데이터 파일 로드
    if not os.path.exists(DATA_FILE):
        print(f"데이터 파일을 찾을 수 없습니다: {DATA_FILE}")
        return
    
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    print(f"총 {len(all_data)} 개의 약물 데이터 로드됨")
    
    # PostgreSQL 연결
    try:
        conn = get_db_connection()
        print("PostgreSQL 연결 성공")
    except Exception as e:
        print(f"PostgreSQL 연결 실패: {e}")
        return
    
    # 데이터 처리
    processed_count = 0
    
    # 처리할 약물 제한 (테스트용)
    limited_data = dict(list(all_data.items())[:5])  # 처음 5개만 처리
    
    for alias, items in limited_data.items():
        print(f"처리 중: {alias} ({len(items)} 개 제품)")
        
        for item in items:
            item_seq = item.get("itemSeq")
            if not item_seq:
                continue
            
            try:
                # 제품 정보 upsert
                upsert_product(conn, item_seq, item)
                
                # 제품 별칭 삽입
                insert_product_alias(conn, alias, item_seq)
                
                # 성분 추출 및 삽입
                ingredients = extract_ingredients(item.get("itemName", ""))
                if ingredients:
                    insert_product_ingredients(conn, item_seq, ingredients)
                
                # 섹션별 텍스트 처리 (임베딩 없이)
                for old_section, new_section in SECTION_MAPPING.items():
                    text = item.get(old_section)
                    if text:
                        # 텍스트 분할
                        text_parts = split_text(text)
                        
                        # PostgreSQL에 섹션 저장
                        upsert_product_sections(conn, item_seq, new_section, text_parts)
                
                processed_count += 1
                
            except Exception as e:
                print(f"제품 {item_seq} 처리 중 오류: {e}")
                continue
        
        # 중간 커밋
        conn.commit()
        print(f"{alias} 완료")
    
    # 최종 커밋
    conn.commit()
    conn.close()
    
    print(f"기본 인제스트 완료! 총 {processed_count} 개 제품 처리됨")

if __name__ == "__main__":
    main() 