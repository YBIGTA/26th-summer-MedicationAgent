#!/usr/bin/env python3
"""
LangChain + Qdrant RAG 데모
약물 정보 검색 및 질의응답을 시연합니다.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

# 환경변수 로드
load_dotenv()

def get_qdrant_client():
    """Qdrant 클라이언트를 반환합니다."""
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

def create_qdrant_vectorstore():
    """Qdrant를 LangChain VectorStore로 변환합니다."""
    client = get_qdrant_client()
    
    # QdrantVectorStore 생성
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="product_sections",
        embeddings=None  # 이미 임베딩된 벡터 사용
    )
    
    return vectorstore

def search_with_filters(query: str, section: str = None, alias: str = None, ingredient: str = None, k: int = 5):
    """필터를 적용하여 검색합니다."""
    client = get_qdrant_client()
    
    # 필터 구축
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
    
    search_filter = Filter(must=conditions) if conditions else None
    
    # 검색 실행
    results = client.search(
        collection_name="product_sections",
        query_vector=None,  # 텍스트 검색은 나중에 구현
        query_filter=search_filter,
        limit=k,
        with_payload=True
    )
    
    return results

def format_search_results(results):
    """검색 결과를 포맷팅합니다."""
    formatted_results = []
    
    for result in results:
        payload = result.payload
        formatted_result = {
            "score": result.score,
            "item_name": payload.get("item_name", "N/A"),
            "section": payload.get("section", "N/A"),
            "aliases": payload.get("aliases", []),
            "ingredients": payload.get("ingredients", []),
            "text": payload.get("text", "N/A") if "text" in payload else "텍스트 없음"
        }
        formatted_results.append(formatted_result)
    
    return formatted_results

def demo_basic_search():
    """기본 검색 데모"""
    print("🔍 기본 검색 데모")
    print("=" * 50)
    
    # 모든 섹션에서 "타이레놀" 검색
    results = search_with_filters("타이레놀", k=3)
    formatted = format_search_results(results)
    
    print(f"타이레놀 관련 결과: {len(formatted)}개")
    for i, result in enumerate(formatted, 1):
        print(f"\n{i}. {result['item_name']}")
        print(f"   섹션: {result['section']}")
        print(f"   별칭: {', '.join(result['aliases'])}")
        print(f"   성분: {', '.join(result['ingredients'])}")
        print(f"   점수: {result['score']:.4f}")

def demo_section_filter():
    """섹션 필터 데모"""
    print("\n🔍 섹션 필터 데모")
    print("=" * 50)
    
    # 상호작용 섹션에서 "와파린" 검색
    results = search_with_filters("와파린", section="interactions", k=3)
    formatted = format_search_results(results)
    
    print(f"와파린 상호작용 관련 결과: {len(formatted)}개")
    for i, result in enumerate(formatted, 1):
        print(f"\n{i}. {result['item_name']}")
        print(f"   섹션: {result['section']}")
        print(f"   별칭: {', '.join(result['aliases'])}")
        print(f"   성분: {', '.join(result['ingredients'])}")

def demo_alias_filter():
    """별칭 필터 데모"""
    print("\n🔍 별칭 필터 데모")
    print("=" * 50)
    
    # 타이레놀의 효능/효과 검색
    results = search_with_filters("효능", alias="타이레놀", section="efficacy", k=3)
    formatted = format_search_results(results)
    
    print(f"타이레놀 효능/효과 관련 결과: {len(formatted)}개")
    for i, result in enumerate(formatted, 1):
        print(f"\n{i}. {result['item_name']}")
        print(f"   섹션: {result['section']}")
        print(f"   별칭: {', '.join(result['aliases'])}")
        print(f"   성분: {', '.join(result['ingredients'])}")

def demo_ingredient_filter():
    """성분 필터 데모"""
    print("\n🔍 성분 필터 데모")
    print("=" * 50)
    
    # 아세트아미노펜 성분의 부작용 검색
    results = search_with_filters("부작용", ingredient="아세트아미노펜", section="side_effects", k=3)
    formatted = format_search_results(results)
    
    print(f"아세트아미노펜 부작용 관련 결과: {len(formatted)}개")
    for i, result in enumerate(formatted, 1):
        print(f"\n{i}. {result['item_name']}")
        print(f"   섹션: {result['section']}")
        print(f"   별칭: {', '.join(result['aliases'])}")
        print(f"   성분: {', '.join(result['ingredients'])}")

def demo_langchain_integration():
    """LangChain 연동 데모"""
    print("\n🤖 LangChain 연동 데모")
    print("=" * 50)
    
    try:
        # QdrantVectorStore 생성
        vectorstore = create_qdrant_vectorstore()
        print("✅ QdrantVectorStore 생성 성공")
        
        # retriever 생성
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 3,
                "filter": {
                    "must": [
                        {"key": "section", "match": {"value": "interactions"}},
                        {"key": "aliases", "match": {"any": ["타이레놀"]}}
                    ]
                }
            }
        )
        print("✅ Retriever 생성 성공")
        
        # 검색 실행
        docs = retriever.get_relevant_documents("와파린 상호작용")
        print(f"✅ 관련 문서 검색 성공: {len(docs)}개")
        
        for i, doc in enumerate(docs, 1):
            print(f"\n{i}. 문서 메타데이터:")
            for key, value in doc.metadata.items():
                print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ LangChain 연동 실패: {e}")

def main():
    """메인 함수"""
    print("💊 약물 정보 RAG 데모")
    print("=" * 60)
    
    # 환경변수 확인
    required_vars = ["QDRANT_URL", "QDRANT_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ 필요한 환경변수가 설정되지 않았습니다: {', '.join(missing_vars)}")
        print("env.sample 파일을 참고하여 .env 파일을 설정하세요.")
        return
    
    print("✅ 환경변수 설정 확인됨")
    
    # 데모 실행
    try:
        demo_basic_search()
        demo_section_filter()
        demo_alias_filter()
        demo_ingredient_filter()
        demo_langchain_integration()
        
        print("\n🎉 모든 데모 완료!")
        
    except Exception as e:
        print(f"❌ 데모 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main() 