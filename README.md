# 💊 Medication Agent - AI 기반 약물 정보 챗봇

AI와 LangChain을 활용한 한국 약물 정보 검색 및 복약 관리 시스템입니다. 사용자가 약물에 대한 질문을 하면 AI가 약물 데이터베이스를 검색하여 정확한 정보를 제공하고, 개인별 복약 일정을 관리할 수 있습니다.

## 🚀 주요 기능

### **1. AI 약물 정보 검색**
- 약물 효능, 용법, 주의사항, 부작용 등 상세 정보 제공
- 약물 간 상호작용 정보 검색
- 자연어로 약물 관련 질문에 답변

### **2. 복약 일정 관리**
- 개인별 복약 체크리스트 생성
- 식전/식후 복용 시간 설정
- 주간 복약 일정 캘린더 생성
- 복약 알림 및 추적

### **3. 모듈화된 아키텍처**
- **config.py**: 애플리케이션 설정 및 환경변수 관리
- **langchain_agent.py**: AI 에이전트 및 벡터 검색 시스템
- **medication_tools.py**: 복약 관리 도구 및 쿼리 분류
- **ui_components.py**: Streamlit UI 컴포넌트 및 세션 관리
- **calendar_utils.py**: 캘린더 생성 및 내보내기 기능
- **medication_db.py**: PostgreSQL 데이터베이스 연동

## 🏗️ 프로젝트 구조

```
📁 Medication Agent
├── 📁 langchain_version/          # LangChain 기반 구현
│   ├── app.py                     # 메인 Streamlit 애플리케이션
│   ├── config.py                  # 애플리케이션 설정 및 상수
│   ├── langchain_agent.py         # LangChain 에이전트 및 도구 관리
│   ├── medication_db.py           # PostgreSQL 데이터베이스 관리
│   ├── medication_tools.py        # 복약 관리 도구 및 쿼리 처리
│   ├── ui_components.py           # Streamlit UI 컴포넌트
│   ├── calendar_utils.py          # 캘린더 유틸리티 함수
│   ├── requirements.txt           # Python 의존성
│   └── README.md                  # LangChain 버전 상세 설명
├── 📁 fastapi_version/            # FastAPI 기반 구현 (향후 추가 예정)
├── 📄 all_drug_data.json          # 한국 약물 정보 데이터베이스
└── 📄 README.md                   # 프로젝트 전체 개요
```

## 📋 모듈별 상세 기능

### **config.py**
- 환경변수 로드 및 애플리케이션 설정 관리
- OpenAI, Qdrant, Supabase 연결 정보
- UI 설정, 프롬프트 템플릿, 에러 메시지 정의

### **langchain_agent.py**
- OpenAI GPT 모델 및 임베딩 초기화
- Qdrant 벡터 데이터베이스 연결 및 검색
- LangChain 에이전트 및 QA 체인 생성
- 약물 정보 검색 도구 구현

### **medication_tools.py**
- 사용자 입력 의도 분류 (추가/조회/수정/삭제/질문/일정)
- 복약 체크리스트 CRUD 작업
- AI 에이전트를 통한 쿼리 처리
- 복약 일정 생성 및 관리

### **ui_components.py**
- Streamlit 세션 상태 관리
- 사이드바, 채팅 인터페이스, 폼 UI 렌더링
- 사용자 인증 및 시스템 초기화
- 메시지 히스토리 관리

### **calendar_utils.py**
- AI 응답에서 캘린더 JSON 파싱
- 주간 복약 일정 이벤트 생성
- iCal 형식 캘린더 내보내기
- 시간대 처리 및 날짜 변환

### **medication_db.py**
- PostgreSQL 데이터베이스 연결 관리
- 사용자 및 복약 정보 CRUD 작업
- Supabase 연동 및 데이터 영속성

## 🔧 기술 스택

### **Backend**
- **Python 3.8+**: 메인 프로그래밍 언어
- **LangChain**: AI 체인 및 도구 관리
- **OpenAI GPT**: 자연어 처리 및 응답 생성
- **Qdrant**: 벡터 데이터베이스 (약물 정보 검색)

### **Frontend**
- **Streamlit**: 웹 인터페이스
- **HTML/CSS**: 사용자 인터페이스 스타일링

### **Database**
- **PostgreSQL**: 사용자 정보 및 복약 데이터 저장
- **Supabase**: 클라우드 데이터베이스 서비스

### **AI/ML**
- **OpenAI Embeddings**: 텍스트 벡터화
- **RAG (Retrieval-Augmented Generation)**: 정확한 정보 검색 및 생성

## 🚀 빠른 시작

### **LangChain 버전 실행**

```bash
# 저장소 클론
git clone [repository-url]

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r langchain_version/requirements.txt

# 환경변수 설정 (.env 파일 생성)
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
OPENAI_API_KEY=your_openai_api_key
SUPABASE_DB_URL=your_supabase_db_url

# 애플리케이션 실행
cd langchain_version
streamlit run app.py
```

## 📱 사용법

### **1. 약물 정보 검색**
- "타이레놀의 효능이 뭔가요?"
- "와파린과 함께 복용하면 안 되는 약이 있나요?"
- "아세트아미노펜의 부작용은?"
- "혈압약 주의사항이 궁금해요"

### **2. 복약 일정 관리**
- 사용자 계정 생성
- 복용 중인 약물 정보 입력
- 식전/식후 복용 시간 설정
- 주간 복약 일정 확인

### **3. 캘린더 내보내기**
- 복약 일정을 iCal 형식으로 내보내기
- 개인 캘린더 앱에 동기화

## 🔍 데이터베이스 스키마

### **Users 테이블**
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### **User Medications 테이블**
```sql
CREATE TABLE user_medications (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    medication_name VARCHAR(200) NOT NULL,
    morning BOOLEAN DEFAULT FALSE,
    lunch BOOLEAN DEFAULT FALSE,
    dinner BOOLEAN DEFAULT FALSE,
    before_meal BOOLEAN DEFAULT FALSE,
    after_meal BOOLEAN DEFAULT FALSE,
    start_date DATE NOT NULL,
    end_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🎯 프로젝트 목표

### **1. 정확한 약물 정보 제공**
- 한국 식약처 공식 약물 정보 활용
- AI 기반 자연어 검색으로 사용자 편의성 향상
- 최신 약물 정보 자동 업데이트

### **2. 개인화된 복약 관리**
- 사용자별 맞춤형 복약 일정
- 복용 시간 알림 및 추적
- 복약 이력 관리

### **3. 의료진과의 소통 지원**
- 복약 정보를 체계적으로 정리
- 의사 상담 시 참고 자료 제공
- 복용 중 이상 반응 기록

## 🤝 기여하기

1. 이 저장소를 포크합니다
2. 새로운 기능 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add some amazing feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## ⚠️ 면책 조항

**중요**: 이 애플리케이션은 교육 및 참고 목적으로만 제공됩니다. 의학적 조언이나 진단을 대체할 수 없습니다. 약물 복용과 관련된 모든 결정은 반드시 의사나 약사와 상의하시기 바랍니다.

## 📞 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해 주세요.

---

**YBIGTA 26기 여름방학 프로젝트**  
