PY=python3
PIP=pip3

.PHONY: install dev run-api run-ui run-chatkit qdrant-up qdrant-down ingest chat eval run-appv2-backend stop-appv2-backend

install:
	$(PIP) install -e .

dev:
	$(PIP) install -e .[dev]

run-api:
	uvicorn app.api:app --reload --port 8000

run-app-frontend:
	cd app/frontend && npm install && npm run dev

run-app-backend:
	 cd app/backend && uv run --active uvicorn app.main:app --reload --port 8002

stop-app-backend:
	 -pkill -f "uvicorn app.main:app --reload --port 8002" || true
	 -lsof -t -i:8002 | xargs kill -9 || true

qdrant-up:
	docker compose -f docker/docker-compose.qdrant.yaml up -d

qdrant-down:
	docker compose -f docker/docker-compose.qdrant.yaml down --remove-orphans

ingest:
	rag ingest --config configs/default.openai.yaml

chat:
	rag chat --config configs/default.openai.yaml

eval:
	rag eval --config configs/default.openai.yaml