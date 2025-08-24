# app/main.py
import os
from fastapi import FastAPI, Depends, Header, HTTPException
from typing import Dict, Any

# 로컬 개발 시 .env 자동 로드
if os.environ.get("ENV", "dev") == "dev":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

app = FastAPI()

def verify_api_key(x_api_key: str = Header(default=None)):
    if x_api_key != os.getenv("PROJECT_API_KEY"):
        raise HTTPException(status_code=401, detail="bad api key")

@app.get("/")
def root():
    return {"ok": True}

@app.get("/healthz")
def healthz():
    return {"ok": True}

# 슬래시 버전도 명시적으로 허용 (리다이렉트 회피)
@app.get("/healthz/")
def healthz_slash():
    return {"ok": True}

@app.get("/diag")
def diag():
    keys = ["SUPABASE_DB_URL","QDRANT_URL","QDRANT_API_KEY","OPENAI_API_KEY","PROJECT_API_KEY"]
    try:
        import psycopg
        psycopg_loaded = True
    except Exception:
        psycopg_loaded = False
    
    return {
        "env_present": {k: bool(os.environ.get(k)) for k in keys},
        "psycopg_loaded": psycopg_loaded,
    }

# --- DB 핑 (지연 임포트) ---
@app.get("/db/ping", dependencies=[Depends(verify_api_key)])
def db_ping():
    try:
        import psycopg
        with psycopg.connect(os.environ["SUPABASE_DB_URL"], connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("select 1")
                return {"ok": cur.fetchone()[0] == 1}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Database connection failed: {str(e)}")

# --- Qdrant 핑 (지연 임포트) ---
@app.get("/qdrant/ping", dependencies=[Depends(verify_api_key)])
def qdrant_ping():
    try:
        from qdrant_client import QdrantClient
        c = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
        colls = c.get_collections().collections
        return {"ok": True, "collections": [x.name for x in colls]}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Qdrant connection failed: {str(e)}")

# --- 임베딩 테스트 (지연 임포트) ---
@app.post("/embed/test", dependencies=[Depends(verify_api_key)])
def embed_test(text: str = "hello world"):
    try:
        from openai import OpenAI
        from qdrant_client.http.models import Distance, VectorParams, PointStruct
        
        oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        emb = oai.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding
        
        from qdrant_client import QdrantClient
        c = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])

        # 콜렉션 생성/재생성
        c.recreate_collection(
            collection_name="smoketest",
            vectors_config=VectorParams(size=len(emb), distance=Distance.COSINE),
        )
        # 업서트
        c.upsert("smoketest", [PointStruct(id=1, vector=emb, payload={"text": text})])
        # 서치
        hits = c.search(collection_name="smoketest", query_vector=emb, limit=1)
        return {"ok": True, "score": float(hits[0].score), "payload": hits[0].payload}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding test failed: {str(e)}")
