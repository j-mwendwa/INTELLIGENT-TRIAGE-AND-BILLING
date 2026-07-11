o.PHONY: install serve ingest lint test-unit test-int test-e2e audit-context prompts-reload clean help

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies + pre-commit hooks
	pip install -e ".[dev,eval]"
	pre-commit install

serve: ## Start the API server (development)
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload --log-level info

ingest: ## Ingest documents from data/raw into the vector store
	python scripts/ingest.py --dir data/raw

ingest-billing: ## Ingest billing knowledge base
	python scripts/ingest.py --dir data/raw/billing --collection billing_kb

ingest-technical: ## Ingest technical knowledge base
	python scripts/ingest.py --dir data/raw/technical --collection technical_kb

ingest-compliance: ## Ingest compliance knowledge base
	python scripts/ingest.py --dir data/raw/compliance --collection compliance_kb

lint: ## Run ruff + mypy
	ruff check src/ tests/
	mypy src/

test-unit: ## Run unit tests (fast, no external calls)
	pytest tests/unit/ -v

test-int: ## Run integration tests (requires GOOGLE_API_KEY)
	pytest -m integration -v

test-e2e: ## Run end-to-end browser tests (requires playwright)
	pytest tests/e2e/ -v

audit-context: ## Audit what the LLM sees for a given message
	@read -p "Enter message: " msg; python scripts/audit_context.py --message "$$msg"

prompts-reload: ## Hot-reload prompts without restarting
	touch configs/config.yaml

eval-rag: ## Run RAG pipeline evaluation
	python evals/run_evals.py --pipeline rag

eval-agent: ## Run full agent graph evaluation
	python evals/run_evals.py --pipeline agent

clean: ## Remove generated artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete; \
	rm -rf .mypy_cache .ruff_cache .pytest_cache

docker-up: ## Start supporting services (Chroma)
	docker compose up -d

docker-down: ## Stop supporting services
	docker compose down
