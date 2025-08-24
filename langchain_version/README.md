# 💊 Medication Agent - LangChain 버전

LangChain만을 사용한 간단한 약물 정보 챗봇

## 🚀 특징

### **기존 버전 vs LangChain 버전**

| 구분 | 기존 버전 | LangChain 버전 |
|------|-----------|----------------|
| **구조** | Streamlit → FastAPI → Qdrant | Streamlit → LangChain → Qdrant |
| **복잡도** | 높음 (3개 컴포넌트) | 낮음 (2개 컴포넌트) |
| **설정** | API 서버 실행 필요 | 단일 앱 실행 |
| **유지보수** | 복잡 | 간단 |

## 🏗️ 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │───►│   LangChain     │───►│     Qdrant      │
│   (Frontend)    │    │   (Backend)     │    │   (Vector DB)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │     OpenAI      │
                       │   (LLM/Embed)   │
                       └─────────────────┘
```

## 🛠️ 설치 및 실행

### 1. 환경 설정
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 환경변수 설정
`.env` 파일 생성:
```bash
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
OPENAI_API_KEY=your_openai_api_key
```

### 3. 실행
```bash
streamlit run app.py
```

## 🔧 LangChain의 장점

### **1. 간단한 구조**
- FastAPI 서버 없이 직접 연결
- 단일 파일로 모든 기능 구현
- 설정 및 배포 간소화

### **2. 내장 기능**
- **RetrievalQA 체인**: RAG 시스템 자동 구성
- **프롬프트 템플릿**: 구조화된 프롬프트 관리
- **메모리 관리**: 대화 히스토리 자동 처리
- **에러 처리**: 내장된 에러 핸들링

### **3. 확장성**
- 다양한 체인 타입 지원
- 커스텀 프롬프트 쉽게 추가
- 메모리 시스템 통합

## 📊 기능 비교

### **기존 버전 (FastAPI)**
```python
# 복잡한 API 호출
response = requests.post(
    f"{api_url}/search",
    headers={"x-api-key": api_key},
    json={"query": query, ...}
)
results = response.json()["results"]
```

### **LangChain 버전**
```python
# 간단한 체인 호출
result = qa_chain({"query": user_input})
ai_response = result["result"]
sources = result.get("source_documents", [])
```

## 🎯 사용법

1. **테스트 모드**: API 없이 테스트 가능
2. **실제 모드**: Qdrant와 OpenAI 연결하여 실제 검색
3. **예시 질문**: 버튼으로 빠른 테스트
4. **대화 히스토리**: 세션별 대화 저장

## 💡 예시 질문

- "타이레놀의 효능이 뭔가요?"
- "와파린과 함께 복용하면 안 되는 약이 있나요?"
- "아세트아미노펜의 부작용은?"
- "타이레놀 복용법 알려주세요"
- "혈압약 주의사항이 궁금해요"

## 🔍 핵심 컴포넌트

### **RetrievalQA 체인**
```python
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)
```

### **프롬프트 템플릿**
```python
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
당신은 한국의 약물 정보 전문가입니다...
참고 정보: {context}
질문: {question}
답변: ...
"""
)
```

## 🚀 장점 요약

1. **단순성**: FastAPI 서버 불필요
2. **빠른 개발**: LangChain 내장 기능 활용
3. **유지보수**: 단일 파일로 모든 기능 관리
4. **확장성**: 다양한 LangChain 컴포넌트 활용 가능
5. **테스트**: 테스트 모드로 API 없이도 개발 가능

## 📝 결론

LangChain만으로도 충분히 강력한 RAG 시스템을 구축할 수 있습니다. FastAPI가 필요한 경우는 다음과 같습니다:

- **다중 클라이언트 지원** (웹, 모바일, 다른 앱)
- **복잡한 비즈니스 로직**
- **마이크로서비스 아키텍처**
- **고성능 API 서버**

하지만 단순한 챗봇이라면 **LangChain 버전이 훨씬 효율적**입니다!
