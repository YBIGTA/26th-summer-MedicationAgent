#!/usr/bin/env python3
"""
약물 데이터 인제스트 스크립트
PostgreSQL과 Qdrant에 약물 데이터를 저장합니다.
"""

import json
import os
import re
from typing import Dict, List, Any
from dotenv import load_dotenv
import psycopg  # psycopg3
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import openai
import uuid

# 환경변수 로드
load_dotenv()

# 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.normpath(os.path.join(BASE_DIR, "..", "all_drug_data.json"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
RECREATE_QDRANT = os.getenv("RECREATE_QDRANT", "false").lower() == "true"

# 섹션 매핑
SECTION_MAPPING = {
    "efcyQesitm": "efficacy",
    "useMethodQesitm": "dosage",
    "atpnWarnQesitm": "warnings",
    "atpnQesitm": "precautions",
    "intrcQesitm": "interactions",
    "seQesitm": "side_effects",
    "depositMethodQesitm": "storage",
}

def get_db_connection():
    """PostgreSQL 연결을 반환합니다. SUPABASE_DB_URL(DSN) 우선."""
    dsn = os.getenv("SUPABASE_DB_URL")
    if not dsn:
        # 호환: 개별 변수로도 시도
        host = os.getenv("PG_HOST")
        db = os.getenv("PG_DB")
        user = os.getenv("PG_USER")
        password = os.getenv("PG_PASSWORD")
        sslmode = os.getenv("PG_SSLMODE", "require")
        if not all([host, db, user]):
            raise RuntimeError("DB 연결 정보가 없습니다. SUPABASE_DB_URL 또는 PG_* 환경변수를 설정하세요.")
        dsn = f"postgresql://{user}:{password or ''}@{host}/{db}?sslmode={sslmode}"
    return psycopg.connect(dsn)

def get_qdrant_client():
    """Qdrant 클라이언트를 반환합니다."""
    return QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

def get_embedding(text: str) -> List[float]:
    """텍스트의 임베딩을 반환합니다."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding

def extract_ingredients(item_name: str) -> List[str]:
    ingredients: List[str] = []
    matches = re.findall(r"\(([^)]+)\)", item_name)
    for match in matches:
        parts = re.split(r"[,·\s]+", match)
        ingredients.extend([p.strip() for p in parts if p.strip()])
    return ingredients

def split_text(text: str, max_length: int = 1000) -> List[str]:
    if not text:
        return []
    if len(text) <= max_length:
        return [text]
    parts: List[str] = []
    current = ""
    for sentence in re.split(r"([.!?]\s+)", text):
        if len(current) + len(sentence) <= max_length:
            current += sentence
        else:
            if current:
                parts.append(current.strip())
            current = sentence
    if current:
        parts.append(current.strip())
    return parts

def upsert_product(conn, item_seq: str, data: Dict[str, Any]):
    with conn.cursor() as cur:
        cur.execute(
            (
                """
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
                """
            ),
            (
                item_seq,
                data.get("entpName"),
                data.get("itemName"),
                data.get("itemImage"),
                data.get("bizrno"),
                (data.get("openDe") or "")[:10] or None,
                (data.get("updateDe") or "")[:10] or None,
                json.dumps(data, ensure_ascii=False),
            ),
        )

def insert_product_alias(conn, alias: str, item_seq: str):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO product_aliases (alias, item_seq)
            VALUES (%s, %s)
            ON CONFLICT (alias, item_seq) DO NOTHING
            """,
            (alias, item_seq),
        )

def insert_product_ingredients(conn, item_seq: str, ingredients: List[str]):
    if not ingredients:
        return
    with conn.cursor() as cur:
        for ing in ingredients:
            cur.execute(
                """
                INSERT INTO product_ingredients (item_seq, ingredient)
                VALUES (%s, %s)
                ON CONFLICT (item_seq, ingredient) DO NOTHING
                """,
                (item_seq, ing),
            )

def upsert_product_sections(conn, item_seq: str, section: str, text_parts: List[str]):
    if not text_parts:
        return
    with conn.cursor() as cur:
        for idx, txt in enumerate(text_parts):
            cur.execute(
                """
                INSERT INTO product_sections (item_seq, section, part_idx, text)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (item_seq, section, part_idx) DO UPDATE SET text = EXCLUDED.text
                """,
                (item_seq, section, idx, txt),
            )

def create_qdrant_collection(client: QdrantClient):
    name = "product_sections"
    if RECREATE_QDRANT:
        try:
            client.delete_collection(name)
            print(f"기존 컬렉션 {name} 삭제")
        except Exception:
            pass
    try:
        client.get_collection(name)
        print(f"컬렉션 {name} 이미 존재")
    except Exception:
        client.create_collection(collection_name=name, vectors_config=VectorParams(size=1536, distance=Distance.COSINE))
        print(f"컬렉션 {name} 생성")

def upsert_qdrant_points(
    client: QdrantClient,
    alias: str,
    item_seq: str,
    section: str,
    text_parts: List[str],
    ingredients: List[str],
    entp_name: str,
    item_name: str,
):
    name = "product_sections"
    for idx, txt in enumerate(text_parts):
        emb = get_embedding(txt)
        payload = {
            "item_seq": item_seq,
            "section": section,
            "part_idx": idx,
            "entp_name": entp_name,
            "item_name": item_name,
            "aliases": [alias],
            "ingredients": ingredients,
            "is_otc": True,
            "update_de": "",
            "text": txt,
        }
        point_id = str(uuid.uuid4())
        client.upsert(collection_name=name, points=[PointStruct(id=point_id, vector=emb, payload=payload)])

def main():
    print("약물 데이터 인제스트 시작...")
    if not os.path.exists(DATA_FILE):
        print(f"데이터 파일을 찾을 수 없습니다: {DATA_FILE}")
        return
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    print(f"총 {len(all_data)} 개 별칭 로드")

    # DB 연결
    try:
        conn = get_db_connection()
        conn.autocommit = True
        print("PostgreSQL 연결 성공")
    except Exception as e:
        print(f"PostgreSQL 연결 실패: {e}")
        return

    # Qdrant
    try:
        qd = get_qdrant_client()
        print("Qdrant 연결 성공")
    except Exception as e:
        print(f"Qdrant 연결 실패: {e}")
        conn.close()
        return

    create_qdrant_collection(qd)

    processed = 0
    for alias, items in all_data.items():
        print(f"처리 중: {alias} ({len(items)} 제품)")
        for item in items:
            item_seq = item.get("itemSeq")
            if not item_seq:
                continue
            try:
                upsert_product(conn, item_seq, item)
                insert_product_alias(conn, alias, item_seq)
                ingredients = extract_ingredients(item.get("itemName", ""))
                insert_product_ingredients(conn, item_seq, ingredients)
                for old, new in SECTION_MAPPING.items():
                    txt = item.get(old)
                    if not txt:
                        continue
                    parts = split_text(txt)
                    upsert_product_sections(conn, item_seq, new, parts)
                    upsert_qdrant_points(qd, alias, item_seq, new, parts, ingredients, item.get("entpName"), item.get("itemName"))
                processed += 1
            except Exception as e:
                print(f"제품 {item_seq} 처리 오류: {e}")
                continue
        try:
            conn.commit()
        except Exception:
            pass
        print(f"{alias} 완료")

    try:
        conn.commit()
    except Exception:
        pass
    conn.close()
    print(f"인제스트 완료: {processed} 제품")

if __name__ == "__main__":
    main() 