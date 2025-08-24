#!/usr/bin/env bash
set -euo pipefail

SERVICE=med-bff
REGION=asia-northeast3
PROJECT=medication-agent-469915

echo "🧪 Running smoke tests for $SERVICE..."

# 프로젝트 번호 가져오기
PN=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
URL="https://${SERVICE}-${PN}.${REGION}.run.app"

echo "📍 Service URL: $URL"
echo ""

# 기본 엔드포인트 테스트
echo "🔍 Testing / endpoint..."
curl -si "$URL/" | head -1

echo "🔍 Testing /healthz endpoint..."
curl -si "$URL/healthz" | head -1

echo "🔍 Testing /diag endpoint..."
curl -s "$URL/diag" | python3 -m json.tool || echo "❌ /diag failed"

echo ""
echo "✅ Smoke tests completed!"
echo "🔍 For detailed testing with API key, use:"
echo "   curl -H 'x-api-key: YOUR_API_KEY' '$URL/db/ping'"
echo "   curl -H 'x-api-key: YOUR_API_KEY' '$URL/qdrant/ping'" 