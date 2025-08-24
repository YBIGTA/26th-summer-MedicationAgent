#!/usr/bin/env python3
"""
약물 검색 API 서버
FastAPI 기반으로 Qdrant를 통한 의미론적 검색을 제공합니다.
"""

import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

# 환경변수 로드
load_dotenv()

app = FastAPI(
    title="Medication Agent API",
    description="약물 정보 의미론적 검색 API",
    version="1.0.0"
)

# 설정
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
GATEWAY_READ_KEY = os.getenv("GATEWAY_READ_KEY", "teammates-read-key")

# Qdrant 클라이언트
def get_qdrant_client():
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

# OpenAI 클라이언트
def get_openai_client():
    return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# API 키 검증
async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != GATEWAY_READ_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# 요청/응답 모델
class SearchRequest(BaseModel):
    query: str
    section: Optional[str] = None
    alias: Optional[str] = None
    ingredient: Optional[str] = None
    k: int = 8

class SearchResult(BaseModel):
    score: float
    item_seq: str
    section: str
    part_idx: int
    item_name: str
    entp_name: str
    aliases: List[str]
    ingredients: List[str]
    is_otc: bool
    update_de: str

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int

def get_embedding(text: str) -> List[float]:
    """텍스트의 임베딩을 반환합니다."""
    client = get_openai_client()
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding

def build_filter(section: Optional[str] = None, 
                alias: Optional[str] = None, 
                ingredient: Optional[str] = None) -> Optional[Filter]:
    """검색 필터를 구축합니다."""
    conditions = []
    
    if section:
        conditions.append(
            FieldCondition(key="section", match=MatchValue(value=section))
        )
    
    if alias:
        conditions.append(
            FieldCondition(key="aliases", match=MatchAny(any=[alias]))
        )
    
    if ingredient:
        conditions.append(
            FieldCondition(key="ingredients", match=MatchAny(any=[ingredient]))
        )
    
    if not conditions:
        return None
    
    return Filter(must=conditions)

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {"message": "Medication Agent API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy"}

@app.post("/search", response_model=SearchResponse)
async def search_medications(
    request: SearchRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    약물 정보를 의미론적으로 검색합니다.
    
    - **query**: 검색할 텍스트
    - **section**: 특정 섹션으로 제한 (efficacy, dosage, warnings, precautions, interactions, side_effects, storage)
    - **alias**: 특정 약물 별칭으로 제한 (타이레놀, 게보린 등)
    - **ingredient**: 특정 성분으로 제한
    - **k**: 반환할 결과 수 (기본값: 8)
    """
    
    try:
        # 쿼리 임베딩 생성
        query_embedding = get_embedding(request.query)
        
        # Qdrant 클라이언트
        qdrant_client = get_qdrant_client()
        
        # 필터 구축
        search_filter = build_filter(
            section=request.section,
            alias=request.alias,
            ingredient=request.ingredient
        )
        
        # 벡터 검색 실행
        search_results = qdrant_client.search(
            collection_name="product_sections",
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=request.k,
            with_payload=True
        )
        
        # 결과 변환
        results = []
        for result in search_results:
            payload = result.payload
            search_result = SearchResult(
                score=result.score,
                item_seq=payload.get("item_seq", ""),
                section=payload.get("section", ""),
                part_idx=payload.get("part_idx", 0),
                item_name=payload.get("item_name", ""),
                entp_name=payload.get("entp_name", ""),
                aliases=payload.get("aliases", []),
                ingredients=payload.get("ingredients", []),
                is_otc=payload.get("is_otc", False),
                update_de=payload.get("update_de", "")
            )
            results.append(search_result)
        
        return SearchResponse(results=results, total=len(results))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 중 오류 발생: {str(e)}")

@app.get("/sections")
async def get_sections(api_key: str = Depends(verify_api_key)):
    """사용 가능한 섹션 목록을 반환합니다."""
    return {
        "sections": [
            "efficacy",      # 효능/효과
            "dosage",        # 용법/용량
            "warnings",      # 주의사항 경고
            "precautions",   # 주의사항
            "interactions",  # 상호작용
            "side_effects",  # 부작용
            "storage"        # 보관법
        ]
    }

@app.get("/aliases")
async def get_aliases(api_key: str = Depends(verify_api_key)):
    """사용 가능한 약물 별칭 목록을 반환합니다."""
    try:
        qdrant_client = get_qdrant_client()
        
        # 모든 별칭 수집
        aliases = set()
        offset = 0
        limit = 100
        
        while True:
            results = qdrant_client.scroll(
                collection_name="product_sections",
                limit=limit,
                offset=offset,
                with_payload=True
            )
            
            if not results[0]:
                break
            
            for result in results[0]:
                payload = result.payload
                if "aliases" in payload:
                    aliases.update(payload["aliases"])
            
            offset += limit
            
            if len(results[0]) < limit:
                break
        
        return {"aliases": sorted(list(aliases))}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"별칭 목록 조회 중 오류 발생: {str(e)}")

@app.get("/ingredients")
async def get_ingredients(api_key: str = Depends(verify_api_key)):
    """사용 가능한 성분 목록을 반환합니다."""
    try:
        qdrant_client = get_qdrant_client()
        
        # 모든 성분 수집
        ingredients = set()
        offset = 0
        limit = 100
        
        while True:
            results = qdrant_client.scroll(
                collection_name="product_sections",
                limit=limit,
                offset=offset,
                with_payload=True
            )
            
            if not results[0]:
                break
            
            for result in results[0]:
                payload = result.payload
                if "ingredients" in payload:
                    ingredients.update(payload["ingredients"])
            
            offset += limit
            
            if len(results[0]) < limit:
                break
        
        return {"ingredients": sorted(list(ingredients))}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"성분 목록 조회 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 