.PHONY: setup smoke test splits features baseline decisions anomaly loao fidelity loop compare figures demo lint fmt clean

setup:    ## install pinned deps into .venv (python 3.11)
	uv sync --extra dev

smoke:    ## the day-one gate: loop runs end-to-end on dummy data
	uv run pytest tests/test_loop_smoke.py -q

test:     ## full suite
	uv run pytest -q

splits:   ## compute + commit the out-of-time boundary and data cards for every real anchor
	uv run python scripts/build_splits.py

features: ## build the feature table over every anchor; record the cost and the coverage
	uv run python scripts/build_features.py

baseline: ## tune the supervised detector on every anchor; commit the reference numbers
	uv run python scripts/build_baseline.py

decisions: ## price the graded action bands and reason codes on every anchor; commit them
	uv run python scripts/build_decisions.py

anomaly:  ## score the zero-day layer against the supervised model on the held-out family
	uv run python scripts/build_anomaly.py

loao:     ## the leave-one-attack-out matrix: every family held out in turn, with the guards
	uv run python scripts/build_loao.py

fidelity: ## harness before generator
	uv run python scripts/build_fidelity.py

loop:     ## System C — the adaptive loop
	uv run python scripts/run_experiment.py experiment=adaptive

compare:  ## the 3-system table: real-only vs SMOTE vs adaptive
	uv run python scripts/run_experiment.py -m experiment=baseline,smote,adaptive

figures:  ## convergence curve + 3-system table from logged runs
	uv run python scripts/make_figures.py

demo:     ## api + streamlit + mlflow
	docker compose up

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

clean:
	rm -rf outputs multirun .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
