#!/usr/bin/env bash
set -euo pipefail

SERVICE=med-bff
REGION=asia-northeast3
PROJECT=medication-agent-469915

echo "🚀 Deploying $SERVICE to Cloud Run..."
echo "Project: $PROJECT"
echo "Region: $REGION"

gcloud run deploy "$SERVICE" \
  --project "$PROJECT" \
  --region "$REGION" \
  --source . \
  --allow-unauthenticated \
  --set-build-env-vars "BP_CPYTHON_VERSION=3.11.9" \
  --update-secrets "PROJECT_API_KEY=PROJECT_API_KEY:latest,OPENAI_API_KEY=OPENAI_API_KEY:latest,SUPABASE_DB_URL=SUPABASE_DB_URL:latest,QDRANT_API_KEY=QDRANT_API_KEY:latest,QDRANT_URL=QDRANT_URL:latest"

echo "✅ Deployment completed!"
echo "🔍 Run 'bash scripts/smoke.sh' to test the deployment" 