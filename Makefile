.PHONY: setup setup-deep smoke test splits features baseline decisions anomaly sequence gnn loao fidelity fidelity-selftest loop table compare figures reproduce claims guardrails demo lint fmt clean

setup:    ## install pinned deps into .venv (python 3.11)
	uv sync --extra dev

setup-deep: ## the same, plus torch and torch-geometric — needed by `make sequence` and
          ## `make gnn` only. Nothing else requires them, and the default suite stays green
          ## without either.
	uv sync --extra dev --extra deep

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

sequence: ## GRU vs LightGBM on the drift arc, sudden and gradual reported apart. Needs
          ## `make setup-deep`; the gate decides whether the model is reported at all.
	uv run python scripts/build_sequence.py

gnn:      ## temporal GNN vs hand-rolled graph features on the mule families, several seeds.
          ## Needs `make setup-deep`; the gate decides which of the two ships.
	uv run python scripts/build_gnn.py

loao:     ## the leave-one-attack-out matrix: every family held out in turn, with the guards
	uv run python scripts/build_loao.py

fidelity: ## the 3-level scorecard on every real anchor; level 3 is the gate, and it can fail
	uv run python scripts/build_fidelity.py

fidelity-selftest: ## prove the harness discriminates, on three cases whose answers are known
	uv run python scripts/build_fidelity.py --selftest

loop:     ## System C — the adaptive loop
	uv run python scripts/run_experiment.py experiment=adaptive

table:    ## THE 3-system table: real-only vs SMOTE vs adaptive, every anchor, every seed
	uv run python scripts/build_three_system.py

compare:  ## the same three systems through the hydra loop on the default config - a pipeline
          ## check, not the reportable table. `make table` is the one that carries numbers.
	uv run python scripts/run_experiment.py -m experiment=baseline,smote,adaptive

figures:  ## convergence curve, realism leash and the numbers behind them, from logged runs
	uv run python scripts/make_figures.py

reproduce: ## THE one command: documents vs artefacts, the wording against the guardrails, then
          ## the whole loop twice on the zero-download default, compared against a committed
          ## expectation. ~2 minutes, nothing to download. Add `--anchor amlworld` to re-run a
          ## seed of the real thing.
	uv run python scripts/reproduce.py

claims:   ## every number quoted in the documents, recomputed from the artefact it names
	uv run python scripts/check_claims.py

guardrails: ## every sentence in the documents, against the seven honesty guardrails — the other
          ## half of the claims audit: `claims` proves the numbers are the artefacts', this
          ## proves the sentences around them are too
	uv run python scripts/check_guardrails.py

demo:     ## api + streamlit + mlflow
	docker compose up

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

clean:
	rm -rf outputs multirun .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
