#!/usr/bin/env python3
"""
💊 Medication Agent - LangChain 버전
LangChain만을 사용한 간단한 약물 정보 챗봇
"""

import streamlit as st
import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from datetime import datetime

# LangChain 관련 import
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

# 환경변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="💊 Medication Agent (LangChain)",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 적용
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        max-height: 600px;
        overflow-y: auto;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        text-align: right;
        max-width: 80%;
        margin-left: auto;
    }
    .bot-message {
        background-color: white;
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        text-align: left;
        max-width: 80%;
        border: 1px solid #ddd;
    }
    .stButton > button {
        width: 100%;
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_langchain():
    """LangChain 컴포넌트 초기화"""
    try:
        # OpenAI 설정
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Qdrant 클라이언트
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        # VectorStore 생성 (공식 문서 기준)
        vectorstore = QdrantVectorStore(
            client=qdrant_client,
            collection_name="product_sections",
            embedding=embeddings
        )
        
        # 프롬프트 템플릿
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
당신은 한국의 약물 정보 전문가입니다. 식약처 공공데이터를 바탕으로 정확하고 도움이 되는 답변을 제공해주세요.

참고 정보:
{context}

질문: {question}

답변: 한국어로 친절하고 정확하게 답변해주세요. 약물의 효능, 용법, 주의사항, 부작용 등을 명확하게 설명해주세요.
"""
        )
        
        # RetrievalQA 체인 생성
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )
        
        # 디버깅: Qdrant에서 직접 검색하여 메타데이터 확인
        try:
            # 샘플 검색으로 메타데이터 구조 확인
            sample_results = qdrant_client.search(
                collection_name="product_sections",
                query_vector=embeddings.embed_query("타이레놀"),
                limit=1,
                with_payload=True
            )
            if sample_results:
                st.write(f"🔍 샘플 검색 결과 메타데이터: {sample_results[0].payload}")
        except Exception as e:
            st.write(f"❌ 샘플 검색 실패: {e}")
        
        # Qdrant 컬렉션 상태 확인
        try:
            collection_info = qdrant_client.get_collection("product_sections")
            st.write(f"📊 Qdrant 컬렉션 정보: {collection_info}")
        except Exception as e:
            st.write(f"❌ Qdrant 컬렉션 확인 실패: {e}")
        
        return qa_chain, vectorstore
        
    except Exception as e:
        st.error(f"LangChain 초기화 실패: {str(e)}")
        return None, None

def get_dummy_data():
    """더미 데이터 (테스트용)"""
    return [
        {
            "score": 0.95,
            "item_name": "타이레놀정500밀리그람(아세트아미노펜)",
            "entp_name": "한국존슨앤드존슨판매(유)",
            "section": "efficacy",
            "aliases": ["타이레놀"],
            "ingredients": ["아세트아미노펜"],
            "text": "이 약은 감기로 인한 발열 및 동통(통증), 두통, 신경통, 근육통, 월경통, 염좌통(삔 통증), 치통, 관절통, 류마티양 동통(통증)에 사용합니다.",
            "update_de": "2024-11-25",
            "is_otc": True
        }
    ]

def check_qdrant_data():
    """Qdrant 데이터 상태 확인"""
    try:
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        # 컬렉션 정보 확인
        collection_info = qdrant_client.get_collection("product_sections")
        st.write(f"📊 컬렉션 정보: {collection_info}")
        
        # 데이터 개수 확인
        scroll_result = qdrant_client.scroll(
            collection_name="product_sections",
            limit=1
        )
        st.write(f"📊 총 데이터 개수: {scroll_result[1]}")
        
        # 샘플 데이터 확인
        if scroll_result[0]:
            sample_point = scroll_result[0][0]
            st.write(f"📋 샘플 데이터 payload: {sample_point.payload}")
            st.write(f"📋 샘플 데이터 vector 길이: {len(sample_point.vector) if sample_point.vector else 'None'}")
            
            # 벡터가 없으면 경고
            if not sample_point.vector:
                st.warning("⚠️ 벡터가 없습니다. DB 초기화가 필요합니다.")
        
        return True
    except Exception as e:
        st.write(f"❌ Qdrant 데이터 확인 실패: {e}")
        return False

def initialize_qdrant_db():
    """Qdrant 데이터베이스 초기화 (컬렉션 재생성)"""
    try:
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        # 기존 컬렉션 삭제
        try:
            qdrant_client.delete_collection("product_sections")
            st.write("🗑️ 기존 컬렉션 삭제됨")
        except Exception as e:
            st.write(f"📝 기존 컬렉션이 없거나 삭제 실패: {e}")
        
        # 새 컬렉션 생성
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
        # JSON 파일에서 데이터 읽기 (상위 폴더에서)
        json_path = "../all_drug_data.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        st.write(f"📊 JSON 데이터 로드됨: {len(data)}개 항목")
        
        # Qdrant 클라이언트
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        # Embeddings 초기화
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 데이터 구조 확인 및 처리
        st.write(f"📊 데이터 타입: {type(data)}")
        if isinstance(data, dict):
            st.write(f"📊 딕셔너리 키: {list(data.keys())}")
            
            # 모든 약물 데이터 수집
            all_items = []
            for drug_name, drug_list in data.items():
                if isinstance(drug_list, list):
                    for drug_item in drug_list:
                        # 약물명을 별칭으로 추가
                        if isinstance(drug_item, dict):
                            drug_item['aliases'] = drug_item.get('aliases', []) + [drug_name]
                        all_items.append(drug_item)
            
            items_to_process = all_items
            st.write(f"📊 처리할 약물 수: {len(items_to_process)}")
        else:
            st.error("❌ 데이터가 딕셔너리 형태가 아닙니다!")
            return False
        
        # 포인트 리스트 초기화
        points = []
            
        for i, item in enumerate(items_to_process):
            try:
                # 각 섹션별로 별도의 포인트 생성
                item_name = item.get('itemName', '')
                
                # 섹션별 데이터 정의
                sections = [
                    ("efficacy", item.get('efcyQesitm', ''), "효능/효과"),
                    ("usage", item.get('useMethodQesitm', ''), "용법/용량"),
                    ("warning", item.get('atpnWarnQesitm', ''), "주의사항 경고"),
                    ("precaution", item.get('atpnQesitm', ''), "주의사항"),
                    ("interaction", item.get('intrcQesitm', ''), "상호작용"),
                    ("side_effect", item.get('seQesitm', ''), "부작용"),
                    ("storage", item.get('depositMethodQesitm', ''), "보관법")
                ]
                
                for section_idx, (section_key, section_text, section_name) in enumerate(sections):
                    # None 값을 빈 문자열로 처리
                    if section_text is None:
                        section_text = ""
                    
                    if section_text.strip():  # 빈 텍스트가 아닌 경우만 처리
                        # 섹션별 텍스트 생성
                        text = f"{item_name} {section_text}"
                        
                        # 임베딩 생성
                        vector = embeddings.embed_query(text)
                        
                        # 포인트 생성 (섹션별로 고유 ID)
                        point_id = i * 10 + section_idx  # 섹션별 고유 ID
                        point = {
                            "id": point_id,
                            "vector": vector,
                            "payload": {
                                "item_name": item_name,
                                "section": section_key,
                                "section_name": section_name,
                                "text": text,
                                "ingredients": item.get("ingredients", []),
                                "aliases": item.get("aliases", [])
                            }
                        }
                        points.append(point)
                
                st.write(f"✅ {i+1}번째 항목 처리됨: {item_name[:30]}... (섹션별로 분리)")
                
            except Exception as e:
                st.write(f"❌ {i+1}번째 항목 처리 실패: {e}")
                st.write(f"📋 실패한 항목 데이터: {item}")
        
        # Qdrant에 업로드
        if points:
            qdrant_client.upsert(
                collection_name="product_sections",
                points=points
            )
            st.success(f"✅ {len(points)}개 항목 업로드 완료!")
        
        return True
    except Exception as e:
        st.error(f"❌ 재인덱싱 실패: {e}")
        return False

def format_source_info(source, index):
    """소스 정보를 포맷팅하는 함수"""
    try:
        # LangChain Document 객체인 경우
        if hasattr(source, 'metadata') and hasattr(source, 'page_content'):
            metadata = source.metadata
            content = source.page_content
            
            # Qdrant에서 직접 데이터 가져오기
            try:
                qdrant_client = QdrantClient(
                    url=os.getenv("QDRANT_URL"),
                    api_key=os.getenv("QDRANT_API_KEY")
                )
                
                # _id가 있으면 해당 포인트 조회
                if '_id' in metadata:
                    point = qdrant_client.retrieve(
                        collection_name="product_sections",
                        ids=[metadata['_id']]
                    )
                    if point:
                        payload = point[0].payload
                        content = payload.get('text', content)
                        item_name = payload.get('item_name', metadata.get('item_name', f'문서 {index}'))
                        section_key = payload.get('section', metadata.get('section', 'N/A'))
                        section_name = payload.get('section_name', 'N/A')
                        # 점수는 메타데이터에서 가져오기 (LangChain에서 제공)
                        score = metadata.get('score', 'N/A')
                        
                        # 섹션 표시 형식 결정
                        if section_name != 'N/A':
                            section = f"{section_key} ({section_name})"
                        else:
                            section = section_key
                    else:
                        item_name = metadata.get('item_name', f'문서 {index}')
                        section_key = metadata.get('section', 'N/A')
                        section_name = metadata.get('section_name', 'N/A')
                        score = metadata.get('score', 'N/A')
                        
                        # 섹션 표시 형식 결정
                        if section_name != 'N/A':
                            section = f"{section_key} ({section_name})"
                        else:
                            section = section_key
                else:
                    # _id가 없으면 메타데이터에서 직접 가져오기
                    item_name = metadata.get('item_name', f'문서 {index}')
                    section_key = metadata.get('section', 'N/A')
                    section_name = metadata.get('section_name', 'N/A')
                    score = metadata.get('score', 'N/A')
                    
                    # 섹션 표시 형식 결정
                    if section_name != 'N/A':
                        section = f"{section_key} ({section_name})"
                    else:
                        section = section_key
                    
            except Exception as e:
                # Qdrant 조회 실패 시 메타데이터에서 가져오기
                item_name = metadata.get('item_name', f'문서 {index}')
                section_key = metadata.get('section', 'N/A')
                section_name = metadata.get('section_name', 'N/A')
                score = metadata.get('score', 'N/A')
                content = content if content else f"Qdrant 조회 실패: {e}"
                
                # 섹션 표시 형식 결정
                if section_name != 'N/A':
                    section = f"{section_key} ({section_name})"
                else:
                    section = section_key
            
            return {
                "title": item_name,
                "section": section,
                "score": score,
                "content": content[:300] + "..." if len(content) > 300 else content,
                "type": "langchain_document"
            }
        # 일반 딕셔너리인 경우
        elif isinstance(source, dict):
            st.write(f"📋 딕셔너리 키: {list(source.keys())}")
            return {
                "title": source.get('item_name', f'문서 {index}'),
                "section": source.get('section', 'N/A'),
                "score": source.get('score', 'N/A'),
                "content": source.get('text', 'N/A')[:300] + "..." if len(source.get('text', '')) > 300 else source.get('text', 'N/A'),
                "type": "dict"
            }
        # 기타 경우
        else:
            st.write(f"📋 기타 타입: {str(source)[:100]}...")
            return {
                "title": f'문서 {index}',
                "section": 'N/A',
                "score": 'N/A',
                "content": str(source)[:300] + "..." if len(str(source)) > 300 else str(source),
                "type": "unknown"
            }
    except Exception as e:
        st.write(f"❌ 포맷팅 오류: {str(e)}")
        return {
            "title": f'문서 {index} (오류)',
            "section": 'N/A',
            "score": 'N/A',
            "content": f"포맷팅 오류: {str(e)}",
            "type": "error"
        }

def initialize_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

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

def main():
    """메인 앱"""
    initialize_session_state()
    
    # 헤더
    st.markdown('<h1 class="main-header">💊 Medication Agent</h1>', unsafe_allow_html=True)
    st.markdown("### 🤖 LangChain 기반 AI 약물 정보 챗봇")
    
    # 사이드바
    with st.sidebar:
        st.header("🔧 설정")
        
        # 테스트 모드 토글
        test_mode = st.checkbox("🧪 테스트 모드", help="API 없이 테스트 데이터로 실행")
        
        # 검색 옵션
        st.subheader("검색 옵션")
        
        # 결과 수 설정
        k = st.slider("📊 검색 결과 수", 1, 10, 3)
        
        st.markdown("---")
        
        # 대화 초기화
        if st.button("🗑️ 대화 초기화", type="secondary"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        
        # 시스템 상태
        st.subheader("📊 시스템 상태")
        
        if test_mode:
            st.success("✅ 테스트 모드 활성화")
        else:
            # LangChain 초기화 상태 확인
            try:
                qa_chain, vectorstore = initialize_langchain()
                if qa_chain:
                    st.success("✅ LangChain 정상")
                else:
                    st.error("❌ LangChain 초기화 실패")
            except Exception as e:
                st.error(f"❌ 연결 오류: {str(e)}")
            
            # Qdrant 데이터 확인 버튼
            if st.button("🔍 Qdrant 데이터 확인", type="secondary"):
                check_qdrant_data()
            
            # DB 초기화 버튼
            if st.button("🗄️ DB 초기화", type="secondary"):
                initialize_qdrant_db()
            
            # 재인덱싱 버튼
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
        # 채팅 컨테이너
        chat_container = st.container()
        
        with chat_container:
            # 기존 메시지 표시
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>나:</strong> {message["content"]}
                        <br><small>{message["timestamp"]}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="bot-message">
                        <strong>💊 AI:</strong> {message["content"]}
                        <br><small>{message["timestamp"]}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 소스 정보 표시
                    if message.get("sources"):
                        with st.expander("📚 참고 정보"):
                            st.write(f"🔍 총 {len(message['sources'])}개의 소스에서 검색됨")
                            
                            for i, source in enumerate(message["sources"][:3], 1):
                                # 디버깅: 원본 소스 정보 출력
                                st.write(f"🔍 원본 소스 {i} 메타데이터: {source.metadata if hasattr(source, 'metadata') else 'No metadata'}")
                                
                                formatted_source = format_source_info(source, i)
                                
                                st.markdown(f"""
                                **{i}. {formatted_source['title']}**
                                - 📂 섹션: {formatted_source['section']}
                                - 📊 점수: {formatted_source['score']}
                                - 🏷️ 타입: {formatted_source['type']}
                                """)
                                
                                # 내용 표시
                                with st.expander(f"📝 내용 보기 ({len(formatted_source['content'])}자)"):
                                    st.text(formatted_source['content'])
                                
                                st.markdown("---")
        
        # 입력 영역
        st.markdown("---")
        
        # 질문 입력 필드
        user_input = st.text_input(
            "💬 질문을 입력하세요...",
            placeholder="예: 타이레놀의 효능이 뭔가요?",
            key="user_input"
        )
        
        # 예시 질문이 있으면 자동으로 전송
        if 'example_question' in st.session_state and st.session_state.example_question:
            user_input = st.session_state.example_question
            # 자동으로 전송 처리
            if user_input.strip():
                # 사용자 메시지 추가
                add_message("user", user_input)
                
                # 응답 생성
                with st.spinner("🔍 정보를 검색하고 있습니다..."):
                    if test_mode:
                        # 테스트 모드: 더미 응답
                        ai_response = f"테스트 모드입니다. '{user_input}'에 대한 답변을 시뮬레이션합니다. 실제로는 LangChain과 Qdrant를 통해 정확한 약물 정보를 검색합니다."
                        sources = get_dummy_data()
                    else:
                        try:
                            # Qdrant에서 직접 검색하여 점수 포함된 결과 가져오기
                            qdrant_client = QdrantClient(
                                url=os.getenv("QDRANT_URL"),
                                api_key=os.getenv("QDRANT_API_KEY")
                            )
                            embeddings = OpenAIEmbeddings(
                                model="text-embedding-3-small",
                                api_key=os.getenv("OPENAI_API_KEY")
                            )
                            
                            # 직접 검색
                            search_results = qdrant_client.search(
                                collection_name="product_sections",
                                query_vector=embeddings.embed_query(user_input),
                                limit=3,
                                with_payload=True
                            )
                            
                            # LangChain Document 형태로 변환
                            sources = []
                            for result in search_results:
                                doc = Document(
                                    page_content=result.payload.get('text', ''),
                                    metadata={
                                        'item_name': result.payload.get('item_name', ''),
                                        'section': result.payload.get('section', ''),
                                        'section_name': result.payload.get('section_name', ''),
                                        'score': f"{result.score:.3f}"  # 점수를 문자열로 변환
                                    }
                                )
                                sources.append(doc)
                            
                            # 직접 검색한 결과로 응답 생성 (LangChain QA 체인 사용하지 않음)
                            if sources:
                                # 검색된 텍스트들을 결합
                                context_text = "\n\n".join([doc.page_content for doc in sources])
                                
                                # OpenAI LLM으로 직접 응답 생성
                                llm = ChatOpenAI(
                                    model="gpt-3.5-turbo",
                                    temperature=0.1,
                                    api_key=os.getenv("OPENAI_API_KEY")
                                )
                                
                                prompt = f"""
당신은 한국의 약물 정보 전문가입니다. 식약처 공공데이터를 바탕으로 정확하고 도움이 되는 답변을 제공해주세요.

참고 정보:
{context_text}

질문: {user_input}

답변: 한국어로 친절하고 정확하게 답변해주세요. 약물의 효능, 용법, 주의사항, 부작용 등을 명확하게 설명해주세요.
"""
                                
                                ai_response = llm.invoke(prompt).content
                            else:
                                ai_response = "죄송합니다. 관련 정보를 찾을 수 없습니다."
                        except Exception as e:
                            ai_response = f"오류가 발생했습니다: {str(e)}"
                            sources = []
                    
                    # AI 메시지 추가
                    add_message("assistant", ai_response, sources)
                
                # 사용 후 삭제
                del st.session_state.example_question
                st.rerun()
        
        # 전송 버튼
        col_a, col_b, col_c = st.columns([1, 1, 1])
        
        with col_b:
            if st.button("🚀 전송", type="primary", use_container_width=True):
                if user_input.strip():
                    # 사용자 메시지 추가
                    add_message("user", user_input)
                    
                    # 응답 생성
                    with st.spinner("🔍 정보를 검색하고 있습니다..."):
                        if test_mode:
                            # 테스트 모드: 더미 응답
                            ai_response = f"테스트 모드입니다. '{user_input}'에 대한 답변을 시뮬레이션합니다. 실제로는 LangChain과 Qdrant를 통해 정확한 약물 정보를 검색합니다."
                            sources = get_dummy_data()
                        else:
                            try:
                                # Qdrant에서 직접 검색하여 점수 포함된 결과 가져오기
                                qdrant_client = QdrantClient(
                                    url=os.getenv("QDRANT_URL"),
                                    api_key=os.getenv("QDRANT_API_KEY")
                                )
                                embeddings = OpenAIEmbeddings(
                                    model="text-embedding-3-small",
                                    api_key=os.getenv("OPENAI_API_KEY")
                                )
                                
                                # 직접 검색
                                search_results = qdrant_client.search(
                                    collection_name="product_sections",
                                    query_vector=embeddings.embed_query(user_input),
                                    limit=3,
                                    with_payload=True
                                )
                                
                                # LangChain Document 형태로 변환
                                sources = []
                                for result in search_results:
                                    doc = Document(
                                        page_content=result.payload.get('text', ''),
                                        metadata={
                                            'item_name': result.payload.get('item_name', ''),
                                            'section': result.payload.get('section', ''),
                                            'section_name': result.payload.get('section_name', ''),
                                            'score': f"{result.score:.3f}"  # 점수를 문자열로 변환
                                        }
                                    )
                                    sources.append(doc)
                                
                                # 직접 검색한 결과로 응답 생성 (LangChain QA 체인 사용하지 않음)
                                if sources:
                                    # 검색된 텍스트들을 결합
                                    context_text = "\n\n".join([doc.page_content for doc in sources])
                                    
                                    # OpenAI LLM으로 직접 응답 생성
                                    llm = ChatOpenAI(
                                        model="gpt-3.5-turbo",
                                        temperature=0.1,
                                        api_key=os.getenv("OPENAI_API_KEY")
                                    )
                                    
                                    prompt = f"""
당신은 한국의 약물 정보 전문가입니다. 식약처 공공데이터를 바탕으로 정확하고 도움이 되는 답변을 제공해주세요.

참고 정보:
{context_text}

질문: {user_input}

답변: 한국어로 친절하고 정확하게 답변해주세요. 약물의 효능, 용법, 주의사항, 부작용 등을 명확하게 설명해주세요.
"""
                                    
                                    ai_response = llm.invoke(prompt).content
                                    
                                    # 소스 정보 확인 (간단한 디버깅)
                                    st.write(f"🔍 검색된 소스 수: {len(sources)}")
                                else:
                                    ai_response = "죄송합니다. 관련 정보를 찾을 수 없습니다."
                            except Exception as e:
                                ai_response = f"오류가 발생했습니다: {str(e)}"
                                sources = []
                        
                        # AI 메시지 추가
                        add_message("assistant", ai_response, sources)
                    
                    # 페이지 새로고침
                    st.rerun()
                else:
                    st.error("질문을 입력해주세요.")
        
        with col_c:
            if st.button("🎲 랜덤 질문", use_container_width=True):
                random_questions = [
                    "타이레놀의 효능이 뭔가요?",
                    "와파린 상호작용에 대해 알려주세요",
                    "아세트아미노펜 부작용은?",
                    "혈압약 주의사항이 궁금해요",
                    "감기약 복용법 알려주세요"
                ]
                import random
                st.session_state.example_question = random.choice(random_questions)
                st.rerun()
    
    with col2:
        st.markdown("### 📈 대화 통계")
        st.metric("총 메시지", len(st.session_state.messages))
        st.metric("사용자 메시지", len([m for m in st.session_state.messages if m["role"] == "user"]))
        st.metric("AI 응답", len([m for m in st.session_state.messages if m["role"] == "assistant"]))
        
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
            if st.button(f"💡 {question}", key=f"recommend_{i}"):
                st.session_state.example_question = question
                st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 🔧 LangChain 장점")
        st.markdown("""
        - **간단한 구조**: FastAPI 없이 직접 연결
        - **내장 프롬프트**: 자동 프롬프트 관리
        - **체인 기반**: 복잡한 로직을 체인으로 구성
        - **메모리 관리**: 대화 히스토리 자동 관리
        """)

if __name__ == "__main__":
    main()
