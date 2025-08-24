#!/usr/bin/env bash
set -euo pipefail

SERVICE=med-bff
REGION=asia-northeast3
PROJECT=medication-agent-469915

echo "📋 Fetching logs for $SERVICE..."

gcloud logging read \
'resource.type="cloud_run_revision" AND resource.labels.service_name="'"$SERVICE"'" AND resource.labels.location="'"$REGION"'"' \
--project "$PROJECT" --freshness=30m --limit 100 --order=desc \
--format='value(timestamp, textPayload, jsonPayload.message)'

echo ""
echo "✅ Logs fetched successfully!"
echo "💡 For real-time logs, use:"
echo "   gcloud logs tail --project=$PROJECT --filter='resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE'" 