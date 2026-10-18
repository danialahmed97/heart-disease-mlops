#!/usr/bin/env bash
# Build the image, deploy to the Docker Desktop (kind) cluster and smoke test.
set -euo pipefail

cd "$(dirname "$0")/.."
IMAGE="heart-disease-api:latest"
NS="heart-disease"
LOCAL_PORT="${LOCAL_PORT:-8080}"
log() { echo "[deploy] $*"; }

log "Checking kubectl context"
kubectl config use-context docker-desktop >/dev/null
kubectl get nodes

log "Building $IMAGE for linux/arm64"
docker build --platform linux/arm64 -t "$IMAGE" .

# Docker Desktop's kind nodes share the local containerd image store. If the
# node is a visible kind container, load the image into it explicitly.
NODE="$(docker ps --format '{{.Names}}' | grep -E 'control-plane' | head -1 || true)"
if [ -n "$NODE" ] && command -v kind >/dev/null; then
  log "Loading image into kind node $NODE"
  kind load docker-image "$IMAGE" --name "${NODE%-control-plane}"
fi

log "Applying manifests"
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml
kubectl rollout restart deployment/heart-disease-api -n "$NS" >/dev/null

log "Waiting for rollout"
if ! kubectl rollout status deployment/heart-disease-api -n "$NS" --timeout=180s; then
  kubectl get pods -n "$NS"; kubectl describe pods -n "$NS" | tail -30
  exit 1
fi
kubectl get pods,svc,hpa -n "$NS"

EXTERNAL="$(kubectl get svc heart-disease-api -n "$NS" \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}{.status.loadBalancer.ingress[0].hostname}')"
BASE=""; PF_PID=""
if [ -n "$EXTERNAL" ] && curl -fsS --max-time 5 "http://${EXTERNAL}/health" >/dev/null 2>&1; then
  BASE="http://${EXTERNAL}"
  log "Using LoadBalancer address $BASE"
else
  # kind LoadBalancer IPs live on Docker's internal network and are usually
  # not routable from macOS, so fall back to a port-forward.
  log "LoadBalancer (${EXTERNAL:-pending}) not reachable from host - using port-forward on :$LOCAL_PORT"
  kubectl port-forward -n "$NS" svc/heart-disease-api "$LOCAL_PORT":80 >/dev/null 2>&1 &
  PF_PID=$!
  trap '[ -n "${PF_PID:-}" ] && kill "$PF_PID" 2>/dev/null || true' EXIT
  BASE="http://localhost:${LOCAL_PORT}"
  for _ in $(seq 1 15); do curl -fsS --max-time 2 "$BASE/health" >/dev/null 2>&1 && break; sleep 1; done
fi

log "GET $BASE/health"
curl -fsS "$BASE/health"; echo
log "POST $BASE/predict"
curl -fsS -X POST "$BASE/predict" -H 'Content-Type: application/json' -d @sample_request.json; echo
log "Deployment verified"
