# 💊 Medication Agent

식약처 공공데이터 기반의 약물 정보 검색 및 질의응답 시스템

## 🚀 주요 기능

- **의미론적 검색**: OpenAI 임베딩을 통한 자연어 기반 약물 정보 검색
- **구조화된 데이터**: PostgreSQL을 통한 관계형 데이터 관리
- **벡터 검색**: Qdrant를 통한 고성능 벡터 유사도 검색
- **RAG 시스템**: LangChain과 연동한 검색 기반 생성 시스템
- **REST API**: FastAPI 기반의 검색 API 제공

## 🏗️ 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   PostgreSQL    │    │     Qdrant      │
│   (검색 API)     │◄──►│   (구조화 데이터) │◄──►│   (벡터 검색)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   Ingest        │    │   LangChain     │
│   (웹 UI)       │    │   (데이터 적재)  │    │   (RAG)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 요구사항

- Python 3.9+
- PostgreSQL
- Qdrant Cloud 또는 자체 설치
- OpenAI API 키

## 🛠️ 설치

### 1. 저장소 클론
```bash
git clone https://github.com/YBIGTA/26th-summer-MedicationAgent.git
cd 26th-summer-MedicationAgent
```

### 2. 환경 설정
```bash
# 환경변수 파일 복사
cp env.sample .env

# .env 파일 편집하여 실제 값 입력
# PostgreSQL, Qdrant, OpenAI 설정
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
```

### 4. PostgreSQL 스키마 생성
```bash
# PostgreSQL에 연결하여 schema.sql 실행
psql -h <host> -U <user> -d <database> -f sql/schema.sql
```

## 🚀 사용법

### 1. 데이터 인제스트
```bash
# 환경변수 설정 후
cd ingest
python ingest_all_json.py
```

### 2. API 서버 실행
```bash
cd api
uvicorn server:app --reload
```

### 3. RAG 데모 실행
```bash
cd examples
python rag_demo.py
```

### 4. Streamlit 앱 실행
```bash
streamlit run app.py
```

## 📊 데이터 구조

### 섹션 매핑
- `efcyQesitm` → `efficacy` (효능/효과)
- `useMethodQesitm` → `dosage` (용법/용량)
- `atpnWarnQesitm` → `warnings` (주의사항 경고)
- `atpnQesitm` → `precautions` (주의사항)
- `intrcQesitm` → `interactions` (상호작용)
- `seQesitm` → `side_effects` (부작용)
- `depositMethodQesitm` → `storage` (보관법)

### 데이터베이스 스키마
- `products`: 제품 기본 정보
- `product_sections`: 섹션별 텍스트 데이터
- `product_aliases`: 제품 별칭
- `product_ingredients`: 제품 성분

## 🔍 API 사용법

### 검색 API
```bash
curl -X POST "http://localhost:8000/search" \
  -H "x-api-key: teammates-read-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "타이레놀 와파린 상호작용",
    "section": "interactions",
    "alias": "타이레놀",
    "k": 5
  }'
```

### 사용 가능한 엔드포인트
- `GET /`: API 정보
- `GET /health`: 헬스 체크
- `POST /search`: 약물 검색
- `GET /sections`: 사용 가능한 섹션 목록
- `GET /aliases`: 사용 가능한 약물 별칭 목록
- `GET /ingredients`: 사용 가능한 성분 목록

## 🔧 환경변수

```bash
# PostgreSQL
PG_HOST=localhost
PG_DB=medication_agent
PG_USER=postgres
PG_PASSWORD=your_password
PG_SSLMODE=require

# Qdrant
QDRANT_URL=https://your-instance.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key

# OpenAI
OPENAI_API_KEY=your_openai_api_key
EMBED_MODEL=text-embedding-3-small

# 게이트웨이
GATEWAY_READ_KEY=teammates-read-key

# Qdrant 재생성
RECREATE_QDRANT=false
```

## 📁 프로젝트 구조

```
26th-summer-MedicationAgent/
├── sql/
│   └── schema.sql              # PostgreSQL 스키마
├── ingest/
│   └── ingest_all_json.py      # 데이터 인제스트 스크립트
├── api/
│   └── server.py               # FastAPI 서버
├── examples/
│   └── rag_demo.py             # LangChain RAG 데모
├── app.py                      # Streamlit 메인 앱
├── agent.py                    # LangChain 에이전트
├── tools.py                    # 검색 도구
├── data.json                   # 기존 약물 데이터
├── all_drug_data.json          # 식약처 공공데이터
├── drug_list.txt               # 약물 목록
├── requirements.txt             # Python 패키지
├── env.sample                  # 환경변수 샘플
└── README.md                   # 프로젝트 문서
```

## 🧪 테스트

### 1. 기본 검색 테스트
```bash
python examples/rag_demo.py
```

### 2. API 테스트
```bash
# 서버 실행 후
curl http://localhost:8000/health
```

### 3. 데이터 적재 확인
```bash
# PostgreSQL에서 데이터 확인
psql -h <host> -U <user> -d <database> -c "SELECT COUNT(*) FROM products;"
```

## 🤝 기여

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 👥 팀

- **YBIGTA 26th Summer Project**
- 약물 정보 검색 및 질의응답 시스템 개발

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해 주세요. 