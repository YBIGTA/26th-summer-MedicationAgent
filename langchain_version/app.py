#!/usr/bin/env python3
"""
�� Medication Agent - LangChain + LangGraph 버전
LangGraph를 사용한 AI 약물 정보 챗봇 + 복약 체크리스트 기능
"""

import os
import json
import re
from typing import List, Dict, Any, TypedDict, Annotated
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from qdrant_client import QdrantClient

# LangGraph imports
from langgraph.graph import StateGraph, END

# 복약 데이터베이스 import
from medication_db import MedicationDatabase

# 환경 변수 로드
load_dotenv()

# CSS 스타일
st.set_page_config(
    page_title="💊 Medication Agent",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 상태 정의
class AgentState(TypedDict):
    """에이전트 상태"""
    user_input: str
    search_results: List[Dict]
    context: str
    ai_response: str
    sources: List[Document]
    error: str
    chat_history: List[Dict]
    conversation_id: str

# 노드 함수들
def search_node(state: AgentState) -> AgentState:
    """검색 노드 - Qdrant에서 관련 정보 검색"""
    try:
        user_input = state["user_input"]
        chat_history = state.get("chat_history", [])
        
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
        
        # 기본 검색
        search_results = qdrant_client.search(
            collection_name="product_sections",
            query_vector=embeddings.embed_query(user_input),
            limit=3,
            with_payload=True
        )
        
        # LangChain Document로 변환
        sources = []
        for result in search_results:
            doc = Document(
                page_content=result.payload.get('text', ''),
                metadata={
                    'item_name': result.payload.get('item_name', ''),
                    'section': result.payload.get('section', ''),
                    'section_name': result.payload.get('section_name', ''),
                    'score': f"{result.score:.3f}"
                }
            )
            sources.append(doc)
        
        # 컨텍스트 생성
        context = "\n\n".join([doc.page_content for doc in sources])
        
        return {
            **state,
            "search_results": search_results,
            "context": context,
            "sources": sources
        }
        
    except Exception as e:
        return {
            **state,
            "error": f"검색 오류: {str(e)}"
        }

def generate_response_node(state: AgentState) -> AgentState:
    """응답 생성 노드 - AI가 답변 생성"""
    try:
        user_input = state["user_input"]
        context = state["context"]
        chat_history = state.get("chat_history", [])
        
        if not context:
            return {
                **state,
                "ai_response": "죄송합니다. 관련 정보를 찾을 수 없습니다."
            }
        
        # OpenAI LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 이전 대화 히스토리를 Message History 형태로 변환
        messages = []
        if chat_history:
            # 최근 4개 대화만 포함
            recent_history = chat_history[-4:]
            
            for msg in recent_history:
                messages.append({"role": "user", "content": msg['user']})
                messages.append({"role": "assistant", "content": msg['assistant']})
        
        # Message History 방식으로 대화 구성
        chat_messages = []
        
        # 시스템 메시지
        system_message = f"""당신은 한국의 약물 정보 전문가입니다. 식약처 공공데이터를 바탕으로 정확하고 도움이 되는 답변을 제공해주세요.

DB내에서 검색한 참고자료:
{context}

매우 중요: 
- "그 약", "이 약", "방금 물어본 약", "지금 우리가 얘기하고 있는 약" 등의 표현이 나오면 이전 대화에서 언급된 약물을 의미합니다.
- 사용자가 약물 이름만 묻는다면, 이전 대화에서 언급된 약물의 정확한 이름만 간단히 답변해주세요.
- 사용자가 약물 정보(효능, 용법, 주의사항, 부작용 등)를 묻는다면, 참고 정보를 바탕으로 상세한 정보를 제공해주세요.
- 질문의 의도를 정확히 파악하여 적절한 수준의 답변을 제공해주세요."""
        
        chat_messages.append({"role": "system", "content": system_message})
        
        # 이전 대화 히스토리 추가
        if messages:
            chat_messages.extend(messages)
        
        # 현재 질문 추가
        chat_messages.append({"role": "user", "content": user_input})
        
        # 응답 생성
        response = llm.invoke(chat_messages)
        ai_response = response.content
        
        return {
            **state,
            "ai_response": ai_response
        }
        
    except Exception as e:
        return {
            **state,
            "error": f"응답 생성 오류: {str(e)}"
        }

def update_chat_history_node(state: AgentState) -> AgentState:
    """채팅 히스토리 업데이트 노드"""
    try:
        user_input = state["user_input"]
        ai_response = state["ai_response"]
        chat_history = state.get("chat_history", [])
        
        # 새로운 대화 추가
        new_message = {
            "user": user_input,
            "assistant": ai_response,
            "timestamp": datetime.now().isoformat()
        }
        
        # 히스토리 업데이트 (최대 10개 대화 유지)
        updated_history = chat_history + [new_message]
        if len(updated_history) > 10:
            updated_history = updated_history[-10:]
        
        return {
            **state,
            "chat_history": updated_history
        }
        
    except Exception as e:
        return {
            **state,
            "error": f"채팅 히스토리 업데이트 오류: {str(e)}"
        }

def error_handler_node(state: AgentState) -> AgentState:
    """오류 처리 노드"""
    error = state.get("error", "")
    if error:
        return {
            **state,
            "ai_response": f"오류가 발생했습니다: {error}"
        }
    return state

# 워크플로우 그래프 생성
def create_workflow() -> StateGraph:
    """LangGraph 워크플로우 생성"""
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("search", search_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("update_chat_history", update_chat_history_node)
    workflow.add_node("error_handler", error_handler_node)
    
    # 엣지 추가
    workflow.add_edge("search", "generate_response")
    workflow.add_edge("generate_response", "update_chat_history")
    workflow.add_edge("update_chat_history", "error_handler")
    workflow.add_edge("error_handler", END)
    
    # 시작점 설정
    workflow.set_entry_point("search")
    
    return workflow.compile()

# 전역 워크플로우 인스턴스
workflow = create_workflow()

def parse_medication_info(user_input):
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
        
        # 복약 정보를 DB에 추가
        if med_db.add_medication(current_user['id'], extracted_info):
            return {
                "success": True,
                "message": f"✅ '{extracted_info['medication_name']}' 복약 정보가 체크리스트에 추가되었습니다!",
                "type": "medication_info",
                "medication": extracted_info
            }
        else:
            return {
                "success": False,
                "message": "이미 존재하는 복약 정보입니다.",
                "type": "medication_info"
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"복약 정보 처리 중 오류가 발생했습니다: {str(e)}",
            "type": "medication_info"
        }

def process_medication_question(user_input: str, conversation_id: str) -> Dict:
    """약물 정보 질문 처리 (LangGraph 사용)"""
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
    
def initialize_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # 복약 체크리스트 관련 상태
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    
    if "parsed_medication" not in st.session_state:
        st.session_state.parsed_medication = {}

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

def process_query(user_input: str, conversation_id: str = None) -> Dict:
    """LangGraph를 사용한 쿼리 처리"""
    existing_history = st.session_state.get("chat_history", [])
    
    initial_state = {
        "user_input": user_input,
        "search_results": [],
        "context": "",
        "ai_response": "",
        "sources": [],
        "error": "",
        "chat_history": existing_history,
        "conversation_id": conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    # 워크플로우 실행
    result = workflow.invoke(initial_state)
    
    return {
        "response": result["ai_response"],
        "sources": result["sources"],
        "error": result.get("error", ""),
        "chat_history": result.get("chat_history", []),
        "conversation_id": result["conversation_id"]
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
        
        collection_info = qdrant_client.get_collection("product_sections")
        st.write(f"📊 컬렉션 정보: {collection_info}")
        
        scroll_result = qdrant_client.scroll(
            collection_name="product_sections",
            limit=1,
            with_vectors=True
        )
        st.write(f"📊 총 데이터 개수: {scroll_result[1]}")
        
        if scroll_result[0]:
            sample_point = scroll_result[0][0]
            st.write(f"📋 샘플 데이터 payload: {sample_point.payload}")
            st.write(f"📋 샘플 데이터 vector 길이: {len(sample_point.vector) if sample_point.vector else 'None'}")
            
            if not sample_point.vector:
                st.warning("⚠️ 벡터가 없습니다. DB 초기화가 필요합니다.")
            else:
                st.success(f"✅ 벡터가 정상적으로 저장되어 있습니다! (길이: {len(sample_point.vector)})")
        
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
        
        try:
            qdrant_client.delete_collection("product_sections")
            st.write("🗑️ 기존 컬렉션 삭제됨")
        except Exception as e:
            st.write(f" 기존 컬렉션이 없거나 삭제 실패: {e}")
        
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
        json_path = "../all_drug_data.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        st.write(f" JSON 데이터 로드됨: {len(data)}개 항목")
        
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        st.write(f"데이터 타입: {type(data)}")
        if isinstance(data, dict):
            st.write(f"딕셔너리 키: {list(data.keys())}")
            
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
        
        uploaded_count = 0
        for i, item in enumerate(items_to_process):
            try:
                if not item.get('efcyQesitm') or not item.get('efcyQesitm').strip():
                    st.write(f"⚠️ {i+1}번째 항목: 빈 텍스트로 건너뜀")
                    continue
                
                sections = {
                    'efficacy': item.get('efcyQesitm', '').strip(),
                    'usage': item.get('useMethodQesitm', '').strip(),
                    'warning': item.get('atpnWarnQesitm', '').strip(),
                    'precaution': item.get('atpnQesitm', '').strip(),
                    'interaction': item.get('intrcQesitm', '').strip(),
                    'side_effect': item.get('seQesitm', '').strip(),
                    'storage': item.get('depositMethodQesitm', '').strip()
                }
                
                for section_name, section_text in sections.items():
                    if section_text and section_text.strip():
                        vector = embeddings.embed_query(section_text)
                        
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
    st.markdown('<h1 class="main-header">�� Medication Agent</h1>', unsafe_allow_html=True)
    st.markdown("### �� LangGraph 기반 AI 약물 정보 챗봇 + 복약 체크리스트")
    
    # 사이드바
    with st.sidebar:
        st.header("🔧 설정")
        
        # 검색 옵션
        st.subheader("검색 옵션")
        k = st.slider("�� 검색 결과 수", 1, 10, 3)
        
        st.markdown("---")
        
        # 사용자 로그인/생성
        st.subheader("�� 사용자 관리")
        
        if not st.session_state.current_user:
            username = st.text_input("사용자명", placeholder="사용자명을 입력하세요")
            email = st.text_input("이메일 (선택사항)", placeholder="이메일을 입력하세요")
            
            if st.button("🔐 로그인/생성", type="primary"):
                if username:
                    user = med_db.get_or_create_user(username, email)
                    if user:
                        st.session_state.current_user = user
                        st.rerun()
                    else:
                        st.error("사용자 생성/조회에 실패했습니다.")
                else:
                    st.error("사용자명을 입력해주세요.")
        else:
            st.success(f"✅ {st.session_state.current_user['name']}으로 로그인됨")
            
            if st.button("🚪 로그아웃"):
                st.session_state.current_user = None
                st.rerun()
            
        st.markdown("---")
        
        # 복약 정보 예시
        st.subheader("�� 복약 정보 예시")
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
                            
                            if st.button("��️ 삭제", key=f"delete_{med['id']}"):
                                if med_db.delete_medication(med['id']):
                                    st.rerun()
            else:
                st.info("�� 복약 정보를 입력하면 자동으로 체크리스트에 추가됩니다!")
        
        st.markdown("---")
        
        # 대화 초기화
        if st.button("🗑️ 대화 초기화", type="secondary"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.rerun()
        
        # 시스템 상태
        st.subheader("⚙️ 시스템 상태")
        
        try:
            qa_chain, vectorstore = initialize_langchain()
            if qa_chain:
                st.success("✅ LangGraph 정상")
            else:
                st.error("❌ LangGraph 초기화 실패")
        except Exception as e:
            st.error(f"❌ 연결 오류: {str(e)}")
        
        if st.button("🔍 Qdrant 데이터 확인", type="secondary"):
            check_qdrant_data()
        
        if st.button("🗄️ DB 초기화", type="secondary"):
            initialize_qdrant_db()
        
        if st.button("🔄 데이터 재인덱싱", type="secondary"):
            reindex_qdrant_data()
        
        st.markdown("---")
        
        # 예시 질문
        st.subheader("💡 예시 질문")
        example_questions = [
            "타이레놀의 효능이 뭔가요?",
            "와파린과 함께 복용하면 안 되는 약이 있나요?",
            "아세트아미노펜의 부작용은?",
            "타이레놀 복용법 알려주세요",
            "혈압약 주의사항이 궁금해요"
        ]
        
        for i, question in enumerate(example_questions):
            if st.button(question, key=f"sidebar_example_{i}"):
                st.session_state.example_question = question
                st.rerun()
    
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
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                if message.get("sources"):
                    st.markdown("---")
                    st.markdown("📚 **참고 정보**")
                    
                    for i, source in enumerate(message["sources"]):
                        with st.expander(f"🔍 소스 {i+1} (점수: {source.metadata.get('score', 'N/A')})"):
                            st.markdown(f"**약물명:** {source.metadata.get('item_name', 'N/A')}")
                            st.markdown(f"**섹션:** {source.metadata.get('section', 'N/A')}")
                            st.markdown(f"**점수:** {source.metadata.get('score', 'N/A')}")
                            
                            with st.expander(f"�� 내용 보기 ({len(source.page_content)}자)"):
                                st.text(source.page_content)
                            
                            st.markdown("---")
        
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
        
        # 예시 질문이 있으면 자동으로 전송
        if 'example_question' in st.session_state and st.session_state.example_question:
            user_input = st.session_state.example_question
            if user_input.strip():
                add_message("user", user_input)
                
                with st.spinner("🤖 스마트 에이전트 분석 중..."):
                    if st.session_state.current_user:
                        result = smart_agent_response(user_input, med_db, st.session_state.current_user, st.session_state.conversation_id)
                    else:
                        result = process_medication_question(user_input, st.session_state.conversation_id)
                    
                    if result["success"]:
                        ai_response = result["message"]
                        sources = result.get("sources", [])
                        
                        if result["type"] == "medication_question":
                            st.session_state.chat_history = result.get("chat_history", [])
                            st.session_state.conversation_id = result.get("conversation_id", st.session_state.conversation_id)
                        else:
                            ai_response = result["message"]
                            sources = []
                    
                    add_message("assistant", ai_response, sources)
                del st.session_state.example_question
                st.rerun()
        
        # 전송 버튼
        col_a, col_b, col_c = st.columns([1, 1, 1])
        
        with col_b:
            if st.button("�� 전송", type="primary", use_container_width=True):
                if user_input.strip():
                    add_message("user", user_input)
                    
                    with st.spinner("🤖 스마트 에이전트 분석 중..."):
                        if st.session_state.current_user:
                            result = smart_agent_response(user_input, med_db, st.session_state.current_user, st.session_state.conversation_id)
                        else:
                            result = process_medication_question(user_input, st.session_state.conversation_id)
                        
                        if result["success"]:
                            ai_response = result["message"]
                            sources = result.get("sources", [])
                            
                            if result["type"] == "medication_question":
                                st.session_state.chat_history = result.get("chat_history", [])
                                st.session_state.conversation_id = result.get("conversation_id", st.session_state.conversation_id)
                            else:
                                ai_response = result["message"]
                                sources = []
                        
                        add_message("assistant", ai_response, sources)
                    st.rerun()
                else:
                    st.error("질문을 입력해주세요.")
        
        with col_c:
            st.write("")  # 빈 공간
    
    with col2:
        st.markdown("### 대화 통계")
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
        
        if st.session_state.chat_history:
            with st.expander("📝 최근 대화 미리보기"):
                for i, msg in enumerate(st.session_state.chat_history[-3:], 1):
                    st.markdown(f"**{i}. 사용자:** {msg['user'][:50]}...")
                    st.markdown(f"**AI:** {msg['assistant'][:50]}...")
                    st.markdown("---")
        
        st.markdown("---")
        
        st.markdown("### 🎯 추천 질문")
        recommended_questions = [
            "타이레놀 효능",
            "와파린 상호작용", 
            "아세트아미노펜 부작용",
            "혈압약 주의사항",
            "감기약 복용법"
        ]
        
        for i, question in enumerate(recommended_questions):
            if st.button(f"{question}", key=f"recommend_{i}"):
                st.session_state.example_question = question
                st.rerun()

if __name__ == "__main__":
    main()