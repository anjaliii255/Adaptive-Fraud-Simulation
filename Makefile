.PHONY: setup smoke test splits fidelity loop compare figures demo lint fmt clean

setup:    ## install pinned deps into .venv (python 3.11)
	uv sync --extra dev

smoke:    ## the day-one gate: loop runs end-to-end on dummy data
	uv run pytest tests/test_loop_smoke.py -q

test:     ## full suite
	uv run pytest -q

splits:   ## compute + commit the out-of-time boundary and data cards for every real anchor
	uv run python scripts/build_splits.py

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
