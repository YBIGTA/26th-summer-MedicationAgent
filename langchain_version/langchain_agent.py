"""
💊 Medication Agent - LangChain 에이전트
AI 에이전트 생성 및 관리를 담당합니다.
"""

import os
import json
from typing import List, Dict, Any, Optional
import streamlit as st

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from qdrant_client import QdrantClient

from config import (
    OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS,
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME,
    DRUG_SEARCH_PROMPT, MEDICATION_SCHEDULE_PROMPT,
    ERROR_MESSAGES, SUCCESS_MESSAGES
)


def initialize_openai() -> Optional[ChatOpenAI]:
    """
    OpenAI 클라이언트를 초기화합니다.
    
    Returns:
        Optional[ChatOpenAI]: OpenAI 클라이언트 또는 None
    """
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.error(ERROR_MESSAGES["openai_connection"])
            return None
        
        return ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE,
            max_tokens=OPENAI_MAX_TOKENS,
            openai_api_key=openai_api_key
        )
    except Exception as e:
        st.error(f"OpenAI 초기화 실패: {e}")
        return None


def initialize_qdrant() -> Optional[QdrantClient]:
    """
    Qdrant 클라이언트를 초기화합니다.
    
    Returns:
        Optional[QdrantClient]: Qdrant 클라이언트 또는 None
    """
    try:
        if not QDRANT_URL or not QDRANT_API_KEY:
            st.error(ERROR_MESSAGES["qdrant_connection"])
            return None
        
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        
        # 연결 테스트
        client.get_collections()
        return client
        
    except Exception as e:
        st.error(f"Qdrant 연결 실패: {e}")
        return None


def initialize_embeddings() -> Optional[OpenAIEmbeddings]:
    """
    OpenAI 임베딩 모델을 초기화합니다.
    
    Returns:
        Optional[OpenAIEmbeddings]: 임베딩 모델 또는 None
    """
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return None
        
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_api_key
        )
    except Exception as e:
        st.error(f"임베딩 모델 초기화 실패: {e}")
        return None


@tool
def search_drug_info(query: str) -> str:
    """
    약물 정보를 검색합니다.
    
    Args:
        query (str): 검색 쿼리
        
    Returns:
        str: 검색 결과
    """
    try:
        # Qdrant 클라이언트 가져오기
        qdrant_client = st.session_state.get("qdrant_client")
        if not qdrant_client:
            return "Qdrant 클라이언트가 초기화되지 않았습니다."
        
        # 임베딩 모델 가져오기
        embeddings = st.session_state.get("embeddings")
        if not embeddings:
            return "임베딩 모델이 초기화되지 않았습니다."
        
        # 쿼리 임베딩 생성
        query_vector = embeddings.embed_query(query)
        
        # 벡터 검색 수행
        search_results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            limit=5
        )
        
        if not search_results:
            return "검색 결과가 없습니다."
        
        # 결과 포맷팅
        formatted_results = []
        for result in search_results:
            payload = result.payload
            formatted_results.append({
                "약물명": payload.get("itemName", "알 수 없음"),
                "효능": payload.get("efcyQesitm", "정보 없음"),
                "용법": payload.get("useMethodQesitm", "정보 없음"),
                "주의사항": payload.get("atpnQesitm", "정보 없음"),
                "부작용": payload.get("seQesitm", "정보 없음"),
                "상호작용": payload.get("intrcQesitm", "정보 없음"),
                "점수": result.score
            })
        
        return json.dumps(formatted_results, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return f"검색 중 오류가 발생했습니다: {e}"


@tool
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
        prompt = MEDICATION_SCHEDULE_PROMPT.format(
            drug_name=drug_name,
            unit=unit,
            schedule_info=schedule_info
        )
        
        # AI 응답 생성
        response = llm.invoke(prompt)
        
        return response.content
        
    except Exception as e:
        return f"일정 생성 중 오류가 발생했습니다: {e}"


def create_agent() -> Optional[AgentExecutor]:
    """
    LangChain 에이전트를 생성합니다.
    
    Returns:
        Optional[AgentExecutor]: 생성된 에이전트 또는 None
    """
    try:
        # OpenAI 클라이언트 초기화
        llm = initialize_openai()
        if not llm:
            return None
        
        # 도구 리스트 정의
        tools = [
            search_drug_info,
            create_medication_schedule
        ]
        
        # 프롬프트 템플릿 생성
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 한국의 약물 정보 전문가입니다. 
            사용자의 질문에 대해 정확하고 이해하기 쉽게 답변해주세요.
            
            사용 가능한 도구:
            - search_drug_info: 약물 정보 검색
            - create_medication_schedule: 복약 일정 생성
            
            한국어로 답변해주세요."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # 에이전트 생성
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        return agent_executor
        
    except Exception as e:
        st.error(f"에이전트 생성 실패: {e}")
        return None


def initialize_qa_chain() -> Optional[Any]:
    """
    질의응답 체인을 초기화합니다.
    
    Returns:
        Optional[Any]: QA 체인 또는 None
    """
    try:
        from langchain.chains import RetrievalQA
        from langchain_qdrant import Qdrant
        
        # Qdrant 클라이언트 가져오기
        qdrant_client = st.session_state.get("qdrant_client")
        if not qdrant_client:
            return None
        
        # 임베딩 모델 가져오기
        embeddings = st.session_state.get("embeddings")
        if not embeddings:
            return None
        
        # OpenAI 클라이언트 가져오기
        llm = st.session_state.get("llm")
        if not llm:
            return None
        
        # Qdrant 벡터스토어 생성
        vectorstore = Qdrant(
            client=qdrant_client,
            collection_name=QDRANT_COLLECTION_NAME,
            embeddings=embeddings
        )
        
        # 프롬프트 템플릿 생성
        prompt_template = ChatPromptTemplate.from_template(DRUG_SEARCH_PROMPT)
        
        # QA 체인 생성
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )
        
        return qa_chain
        
    except Exception as e:
        st.error(f"QA 체인 초기화 실패: {e}")
        return None


def check_qdrant_data() -> bool:
    """
    Qdrant에 약물 데이터가 있는지 확인합니다.
    
    Returns:
        bool: 데이터 존재 여부
    """
    try:
        qdrant_client = st.session_state.get("qdrant_client")
        if not qdrant_client:
            return False
        
        # 컬렉션 정보 확인
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if QDRANT_COLLECTION_NAME not in collection_names:
            return False
        
        # 컬렉션 내 데이터 수 확인
        collection_info = qdrant_client.get_collection(QDRANT_COLLECTION_NAME)
        return collection_info.points_count > 0
        
    except Exception:
        return False


def initialize_qdrant_db() -> bool:
    """
    Qdrant 데이터베이스를 초기화합니다.
    
    Returns:
        bool: 초기화 성공 여부
    """
    try:
        qdrant_client = st.session_state.get("qdrant_client")
        if not qdrant_client:
            return False
        
        # 컬렉션 생성
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config={
                "size": 1536,  # OpenAI text-embedding-3-small 차원
                "distance": "Cosine"
            }
        )
        
        st.success(SUCCESS_MESSAGES["qdrant_initialized"])
        return True
        
    except Exception as e:
        st.error(f"Qdrant 초기화 실패: {e}")
        return False


def reindex_qdrant_data() -> bool:
    """
    약물 데이터를 Qdrant에 재인덱싱합니다.
    
    Returns:
        bool: 인덱싱 성공 여부
    """
    try:
        qdrant_client = st.session_state.get("qdrant_client")
        embeddings = st.session_state.get("embeddings")
        
        if not qdrant_client or not embeddings:
            return False
        
        # 약물 데이터 파일 읽기
        drug_data_path = os.path.join(os.path.dirname(__file__), "..", "all_drug_data.json")
        if not os.path.exists(drug_data_path):
            st.error("약물 데이터 파일을 찾을 수 없습니다.")
            return False
        
        with open(drug_data_path, 'r', encoding='utf-8') as f:
            drug_data = json.load(f)
        
        # 데이터 포인트 생성
        points = []
        for drug_name, drug_list in drug_data.items():
            for drug in drug_list:
                # 약물 정보를 하나의 텍스트로 결합
                text_content = f"""
                약물명: {drug.get('itemName', '')}
                효능: {drug.get('efcyQesitm', '')}
                용법: {drug.get('useMethodQesitm', '')}
                주의사항: {drug.get('atpnQesitm', '')}
                부작용: {drug.get('seQesitm', '')}
                상호작용: {drug.get('intrcQesitm', '')}
                보관법: {drug.get('depositMethodQesitm', '')}
                """
                
                # 임베딩 생성
                vector = embeddings.embed_query(text_content)
                
                points.append({
                    "id": f"{drug.get('itemSeq', '')}_{hash(drug_name)}",
                    "vector": vector,
                    "payload": {
                        "drug_name": drug_name,
                        "itemName": drug.get('itemName', ''),
                        "efcyQesitm": drug.get('efcyQesitm', ''),
                        "useMethodQesitm": drug.get('useMethodQesitm', ''),
                        "atpnQesitm": drug.get('atpnQesitm', ''),
                        "seQesitm": drug.get('seQesitm', ''),
                        "intrcQesitm": drug.get('intrcQesitm', ''),
                        "depositMethodQesitm": drug.get('depositMethodQesitm', ''),
                        "text_content": text_content
                    }
                })
        
        # 데이터 업로드
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=points
        )
        
        st.success(SUCCESS_MESSAGES["data_indexed"])
        return True
        
    except Exception as e:
        st.error(f"데이터 인덱싱 실패: {e}")
        return False
