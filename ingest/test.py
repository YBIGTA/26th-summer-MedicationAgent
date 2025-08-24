import os
import json
import subprocess
import requests

try:
	from dotenv import load_dotenv, find_dotenv
	# 현재 작업 디렉토리 기준으로 상위 경로까지 .env 탐색
	env_path = find_dotenv(usecwd=True)
	if env_path:
		load_dotenv(env_path)
except Exception:
	pass

# 우선순위: GCLOUD_URL(요청사항) > BFF_URL(이전 키) > gcloud로 계산
SERVICE = os.getenv("BFF_SERVICE", "med-bff")
REGION = os.getenv("BFF_REGION", "asia-northeast3")
PROJECT = os.getenv("GCP_PROJECT", "medication-agent-469915")
URL = os.getenv("GCLOUD_URL") or os.getenv("BFF_URL")
API = os.getenv("PROJECT_API_KEY")

if not URL:
	try:
		pn = subprocess.check_output([
			"gcloud", "projects", "describe", PROJECT, "--format=value(projectNumber)",
		], text=True).strip()
		URL = f"https://{SERVICE}-{pn}.{REGION}.run.app"
	except Exception as e:
		raise SystemExit(f"URL 미설정이며 자동 계산 실패: {e}")

headers = {"x-api-key": API} if API else {}

def dump(resp):
	print(resp.status_code, resp.headers.get("content-type"))
	try:
		print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
	except Exception:
		print(resp.text)
	print("-" * 60)

print("URL=", URL)
print("API set=", bool(API))
print("-- GET /")
dump(requests.get(f"{URL}/"))

print("-- GET /diag")
dump(requests.get(f"{URL}/diag"))

if API:
	print("-- GET /db/ping")
	dump(requests.get(f"{URL}/db/ping", headers=headers))
	print("-- GET /qdrant/ping")
	dump(requests.get(f"{URL}/qdrant/ping", headers=headers))
else:
	print("PROJECT_API_KEY가 없어 보호 엔드포인트는 생략합니다.")

# --- 의미검색 Top-5 스모크 (와파린+타이레놀) ---
try:
	from openai import OpenAI
	from qdrant_client import QdrantClient
	from qdrant_client.http import models as qm
	OAI_KEY = os.getenv("OPENAI_API_KEY")
	QURL = os.getenv("QDRANT_URL")
	QKEY = os.getenv("QDRANT_API_KEY")
	if OAI_KEY and QURL:
		print("-- SEMANTIC SEARCH (와파린 + 타이레놀 | section=interactions) --")
		client = OpenAI(api_key=OAI_KEY)
		q = "와파린 복용 중인데 타이레놀 같이 먹어도 돼?"
		emb = client.embeddings.create(model=os.getenv("EMBED_MODEL", "text-embedding-3-small"), input=q).data[0].embedding
		qc = QdrantClient(url=QURL, api_key=QKEY)
		res = qc.search(
			collection_name="product_sections",
			query_vector=emb,
			limit=5,
			query_filter=qm.Filter(must=[qm.FieldCondition(key="section", match=qm.MatchValue(value="interactions"))])
		)
		for i, p in enumerate(res, 1):
			md = p.payload
			print(f"{i}. score={p.score:.3f}")
			print(f"   item_name: {md.get('item_name')} | item_seq: {md.get('item_seq')} | section: {md.get('section')}")
			snip = (md.get("text") or "").replace("\n", " ")[:220]
			print("   snippet:", snip)
			print("---")
	else:
		print("OPENAI_API_KEY 또는 QDRANT_URL이 없어 의미검색을 생략합니다.")
except Exception as e:
	print("의미검색 수행 중 오류:", e) 