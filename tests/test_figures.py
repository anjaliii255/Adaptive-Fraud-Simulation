"""▲ A — the figures come from logged runs, and the same log always draws the same picture.

A figure is the one artefact that travels furthest from the run that made it, so the two things
worth testing are that it is reproducible and that it refuses to invent data it does not have.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "make_figures", Path(__file__).resolve().parents[1] / "scripts" / "make_figures.py"
)
make_figures = importlib.util.module_from_spec(_SPEC)
sys.modules["make_figures"] = make_figures
_SPEC.loader.exec_module(make_figures)


def _round(i: int, evasion: float, recall: float, penalty: float = 0.1) -> dict:
    return {
        "round": i,
        "rejected_by_audit": False,
        "evasion_rate": evasion,
        "recall_at_fixed_fpr": recall,
        "pr_auc": 0.05,
        "realism_penalty": penalty,
        "fitness": evasion - 0.5 * penalty,
    }


def _blob(n_seeds: int = 3, n_rounds: int = 4) -> dict:
    return {
        "anchor": "amlworld",
        "typology": "GATHER-SCATTER",
        "split_digest": "deadbeef",
        "operating_point": {"fixed_fpr": 0.01, "k": 100},
        "rounds": n_rounds,
        "runs": [
            {
                "seed": 1000 + s,
                "positives": 173,
                "base_rate": 0.0005,
                "convergence": [
                    _round(i, 0.9 - 0.1 * i - 0.01 * s, 0.5 + 0.02 * s) for i in range(n_rounds)
                ],
            }
            for s in range(n_seeds)
        ],
    }


def test_regenerating_from_the_same_log_produces_identical_bytes(tmp_path):
    """The reproducibility claim is only worth making if the figure itself is byte-stable."""
    blob = _blob()
    first = make_figures.abcd_convergence_figure(blob, tmp_path / "a.png").read_bytes()
    second = make_figures.abcd_convergence_figure(blob, tmp_path / "b.png").read_bytes()
    assert first == second


def test_the_numbers_beside_the_figure_carry_every_seed(tmp_path):
    """A curve nobody can check against its values is a drawing, not a result."""
    blob = _blob(n_seeds=3)
    text = make_figures.abcd_numbers(blob, tmp_path / "n.md").read_text()
    for run in blob["runs"]:
        assert str(run["seed"]) in text
    assert "deadbeef" in text  # the split the numbers belong to
    assert "rejected 0 of 12 rounds" in text


def test_a_short_run_clips_the_mean_instead_of_padding_it(tmp_path):
    """One seed that died early must not be silently extended to the length of the others."""
    blob = _blob(n_seeds=2, n_rounds=4)
    blob["runs"][0]["convergence"] = blob["runs"][0]["convergence"][:2]
    traces = make_figures._traces(blob["runs"], "evasion_rate")
    assert [len(t) for t in traces] == [2, 2]


def test_a_run_with_no_per_round_trace_is_skipped_not_drawn(tmp_path, capsys):
    """Better no figure than a figure of an empty loop."""
    blob = _blob()
    for run in blob["runs"]:
        run["convergence"] = []
    (tmp_path / "empty.json").write_text(__import__("json").dumps(blob))
    assert make_figures.build_abcd_figures(tmp_path) == []
    assert "skipping" in capsys.readouterr().out


def test_the_pipeline_check_banner_survives_into_the_image():
    """A synthetic-data figure has to carry its caveat, because it outlives the terminal."""
    fig = make_figures.plt.figure()
    make_figures.stamp(fig, pipeline_check=True, data_name="synthetic")
    assert any(make_figures.BANNER in t.get_text() for t in fig.texts)
    make_figures.plt.close(fig)


@pytest.mark.parametrize("field", ["evasion_rate", "recall_at_fixed_fpr", "realism_penalty"])
def test_every_plotted_series_is_read_from_the_log_not_recomputed(field):
    """The figures may only draw fields the run actually wrote."""
    blob = _blob()
    assert all(field in c for r in blob["runs"] for c in r["convergence"])
    assert len(make_figures._mean(make_figures._traces(blob["runs"], field))) == blob["rounds"]
