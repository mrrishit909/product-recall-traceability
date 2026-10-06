export DATABASE_URL ?= postgresql://app:app-dev-only@127.0.0.1:55380/app
PY = ./venv/bin/python
BASE = http://127.0.0.1:8380

bootstrap:      ## local virtualenv + database container
	python3 -m venv venv && ./venv/bin/pip install -q -r requirements-dev.txt
	docker compose up -d db
seed:           ## migrate and load the synthetic world (no-op if already seeded)
	$(PY) -m rc.seed --if-empty
dev:            ## API with reload on :8380, jobs run by a local worker
	($(PY) -m core.jobs rc.api &) && $(PY) -m uvicorn rc.api:app --reload --port 8380
test:
	$(PY) -m pytest --cov=rc --cov=core --cov-report=term-missing
demo:           ## the full stack in Docker, then the demo scenario against it
	docker compose up --build -d --wait && $(PY) -m core.scenario $(BASE)
load-test:      ## run `make demo` first
	$(PY) -m core.loadtest $(BASE) arden-viewer-demo 32 10 "GET /v1/recalls/3f888ae6-1aff-5a13-9623-4df87a959386/impact" "GET /v1/units/ADU-027069/genealogy" "GET /v1/plant" 
evaluate:       ## held-out plants -> docs/evaluation.md
	$(PY) -m rc.evaluate > docs/evaluation.md
reset:          ## drop all data (volume included)
	docker compose down -v
.PHONY: bootstrap seed dev test demo load-test evaluate reset
