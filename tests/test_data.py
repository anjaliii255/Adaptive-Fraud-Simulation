"""■ B — real data enters the contract clean, and the split it is measured on cannot drift.

Ticket 02's two failure modes both look like success until much later:

  * a real row that picks up a `vector_id` has gained a label path, and the leave-one-attack-out
    carve-out will happily hold it out of training as though it were a synthetic family;
  * a split boundary re-derived per run moves whenever the pool composition moves, so two runs
    are compared at two different partitions and nobody notices.

The tests that need the real files skip cleanly when they are absent — `data/**` is gitignored
and a fresh clone has nothing to download.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import yaml

from afl.contract.schema import Rail, Transaction
from afl.data import loaders
from afl.data.splits import CommittedSplit, out_of_time_split

CONFIG_DIR = Path("config/data")
SPLIT_DIR = Path("artifacts/splits")
CARD_DIR = Path("docs/data-cards")

T0 = datetime(2023, 1, 1)


def real_anchors() -> list[dict]:
    """Every data config that names a loader — the real anchors, not the synthetic default."""
    configs = [yaml.safe_load(p.read_text()) for p in sorted(CONFIG_DIR.glob("*.yaml"))]
    return [c for c in configs if c.get("loader")]


ANCHORS = real_anchors()
ANCHOR_IDS = [c["name"] for c in ANCHORS]


def anchor_files_present(cfg: dict) -> bool:
    root = Path(cfg["source"].get("place_in", "data/raw"))
    name = cfg["source"].get("transactions_file") or cfg["source"].get("file")
    return (root / name).exists()


def txns(n: int, start: datetime = T0, fraud_every: int = 10) -> list[Transaction]:
    return [
        Transaction(
            txn_id=f"t{i:05d}",
            ts=start + timedelta(hours=i),
            src=f"s{i % 7}",
            dst=f"d{i % 5}",
            amount=100.0 + i,
            rail=Rail.A2A,
            is_fraud=i % fraud_every == 0,
        )
        for i in range(n)
    ]


# ── real rows carry no provenance ───────────────────────────────────────────────
def test_provenance_guard_catches_a_real_row_wearing_a_vector_id():
    rows = txns(5)
    rows[3] = rows[3].model_copy(update={"vector_id": "S1"})
    with pytest.raises(AssertionError, match="provenance is for synthetic rows only"):
        loaders.assert_no_provenance(rows, "paysim")


def test_provenance_guard_catches_an_attack_run_id_too():
    rows = txns(3)
    rows[0] = rows[0].model_copy(update={"attack_run_id": "run-7"})
    with pytest.raises(AssertionError, match="attack_run_id"):
        loaders.assert_no_provenance(rows, "amlsim")


def test_a_clean_batch_passes_the_guard():
    loaders.assert_no_provenance(txns(20), "paysim")


# ── sampling keeps entities whole ───────────────────────────────────────────────
def test_entity_bucket_is_stable_across_processes():
    """crc32, not `hash()` — a salted hash would resample the dataset on every interpreter."""
    ids = pd.Series(["acct-1", "acct-2", "acct-3"])
    assert loaders.entity_bucket(ids).tolist() == loaders.entity_bucket(ids).tolist()
    assert loaders.entity_bucket(pd.Series(["acct-1"])).tolist() == [
        loaders.entity_bucket(ids).tolist()[0]
    ]


def test_sampling_keeps_whole_entities_never_partial_histories():
    """Half an account's history is a velocity profile no production scorer would ever see."""
    df = pd.DataFrame({"dst": [f"d{i % 40}" for i in range(4_000)], "amount": range(4_000)})
    sampled = loaders.sample_by_entity(df, "dst", 0.5)

    kept = set(sampled["dst"])
    assert 0 < len(kept) < 40, "the sample should drop some entities and keep others"
    for entity in kept:
        assert (sampled["dst"] == entity).sum() == (df["dst"] == entity).sum()


def test_sampling_is_deterministic():
    df = pd.DataFrame({"dst": [f"d{i}" for i in range(1_000)]})
    first = loaders.sample_by_entity(df, "dst", 0.3)
    second = loaders.sample_by_entity(df, "dst", 0.3)
    assert first.index.tolist() == second.index.tolist()


def test_a_full_sample_is_a_no_op():
    df = pd.DataFrame({"dst": ["a", "b", "c"]})
    assert loaders.sample_by_entity(df, "dst", 1.0).equals(df)


@pytest.mark.parametrize("fraction", [0.0, -0.1])
def test_a_nonsense_sample_fraction_is_rejected(fraction):
    """A fraction at or below zero would silently return an empty dataset."""
    with pytest.raises(ValueError, match="sample fraction"):
        loaders.sample_by_entity(pd.DataFrame({"dst": ["a", "b"]}), "dst", fraction)


def test_steps_become_timestamps_off_a_fixed_epoch():
    import numpy as np

    stamps = loaders.steps_to_timestamps(np.array([0, 1, 24]), "hours", T0)
    assert pd.Timestamp(stamps[0]).to_pydatetime() == T0
    assert pd.Timestamp(stamps[2]).to_pydatetime() == T0 + timedelta(hours=24)
    with pytest.raises(ValueError, match="unknown time unit"):
        loaders.steps_to_timestamps(np.array([1]), "fortnights", T0)


# ── the committed boundary ──────────────────────────────────────────────────────
def a_split(**kw) -> CommittedSplit:
    base = dict(
        dataset="test",
        train_end=T0 + timedelta(hours=100),
        test_start=T0 + timedelta(hours=125),
        embargo_rationale="velocity windows look back 24h",
    )
    return CommittedSplit(**{**base, **kw})


def test_a_zero_embargo_is_not_an_embargo():
    with pytest.raises(ValueError, match="embargo must be non-zero"):
        a_split(test_start=T0 + timedelta(hours=100))


def test_an_embargo_with_no_rationale_is_a_magic_number():
    with pytest.raises(ValueError, match="rationale"):
        a_split(embargo_rationale="   ")


def test_split_round_trips_through_json():
    split = a_split(train_end_step=100, test_start_step=125, stats={"full": {"rows": 10}})
    restored = CommittedSplit.from_dict(json.loads(json.dumps(split.to_dict())))
    assert restored == split
    assert restored.digest == split.digest


def test_a_hand_edited_boundary_is_caught_by_its_own_digest():
    raw = a_split().to_dict()
    # a plausible-looking edit: still a valid boundary, just not the one that was committed
    raw["train_end"] = (T0 + timedelta(hours=90)).isoformat()
    with pytest.raises(ValueError, match="does not match the boundary"):
        CommittedSplit.from_dict(raw)


def test_an_old_artefact_version_fails_loudly():
    raw = a_split().to_dict()
    raw["version"] = 0
    with pytest.raises(ValueError, match="version"):
        CommittedSplit.from_dict(raw)


def test_the_embargo_window_is_dropped_not_assigned():
    split, rows = a_split(), txns(200)
    train, test = split.apply(rows)
    assert train and test
    assert len(train) + len(test) < len(rows)
    assert max(t.ts for t in train) < min(t.ts for t in test)
    assert min(t.ts for t in test) - max(t.ts for t in train) >= split.embargo


def test_applying_the_same_boundary_twice_gives_the_same_partition():
    split, rows = a_split(), txns(200)
    assert [t.txn_id for t in split.apply(rows)[0]] == [t.txn_id for t in split.apply(rows)[0]]


def test_the_committed_boundary_does_not_move_when_the_pool_grows():
    """The whole reason the boundary is committed rather than re-derived per run.

    A fraction splits at 70% of *whatever it was handed*. Add a batch of attack rows, or change
    the sample fraction, and last week's number is measured against a different partition than
    this week's — with nothing in the diff to show it.
    """
    rows = txns(200)
    extra = [
        t.model_copy(update={"txn_id": f"x{i}", "ts": t.ts + timedelta(hours=200)})
        for i, t in enumerate(txns(100))
    ]
    split = a_split()

    committed_before = {t.txn_id for t in split.apply(rows)[0]}
    committed_after = {t.txn_id for t in split.apply(rows + extra)[0]}
    assert committed_before == committed_after & {t.txn_id for t in rows}

    frac_before = {t.txn_id for t in out_of_time_split(rows)[0]}
    frac_after = {t.txn_id for t in out_of_time_split(rows + extra)[0]}
    assert frac_before != frac_after, "if this ever passes, the committed split is redundant"


def test_split_saves_and_loads_from_disk(tmp_path):
    split = a_split()
    path = split.save(tmp_path)
    assert path.exists()
    assert CommittedSplit.load("test", tmp_path) == split


def test_a_missing_split_artefact_names_the_command_that_makes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_splits.py"):
        CommittedSplit.load("nope", tmp_path)


# ── the committed artefacts on disk ─────────────────────────────────────────────
@pytest.mark.parametrize("cfg", ANCHORS, ids=ANCHOR_IDS)
def test_every_real_anchor_has_a_committed_split_and_a_data_card(cfg):
    """These are repo artefacts, not run artefacts: they are committed and read, never rebuilt."""
    name = cfg["name"]
    split = CommittedSplit.load(name, SPLIT_DIR)

    assert split.embargo > timedelta(0), "the embargo gap must be non-zero"
    assert len(split.embargo_rationale.split()) > 5, "the rationale must actually say something"
    assert split.stats["full"]["rows"] > 0
    assert 0.0 < split.stats["full"]["base_rate"] < 1.0
    assert (CARD_DIR / f"{name}.md").exists(), f"no data card for {name}"


@pytest.mark.parametrize("cfg", ANCHORS, ids=ANCHOR_IDS)
def test_the_data_card_records_what_the_ticket_asks_it_to(cfg):
    card = (CARD_DIR / f"{cfg['name']}.md").read_text().lower()
    for required in ("source", "licence", "base rate", "embargo", "quirks", "cannot tell us"):
        assert required in card, f"{cfg['name']} data card is missing {required!r}"


@pytest.mark.parametrize("cfg", ANCHORS, ids=ANCHOR_IDS)
def test_the_real_base_rate_is_reported_against_the_synthetic_default(cfg):
    """Every operating point is a function of the base rate, so the gap is reported, not implied."""
    stats = CommittedSplit.load(cfg["name"], SPLIT_DIR).stats
    assert stats["synthetic_base_rate"] > 0
    ratio = stats["base_rate_ratio_vs_synthetic"]
    assert ratio is not None
    if ratio <= 0.1 or ratio >= 10:  # an order of magnitude apart: it has to be said out loud
        assert "order of magnitude" in (CARD_DIR / f"{cfg['name']}.md").read_text().lower()


# ── the synthetic default still needs no download ───────────────────────────────
def test_the_synthetic_default_loads_nothing_and_raises_nothing():
    cfg = yaml.safe_load((CONFIG_DIR / "synthetic.yaml").read_text())
    assert cfg.get("loader") is None
    assert loaders.load_from_config(cfg) == []


def test_an_unknown_dataset_name_is_a_loud_error():
    with pytest.raises(KeyError, match="unknown dataset"):
        loaders.load("banksim")


def test_a_missing_download_says_where_to_get_it(tmp_path):
    with pytest.raises(loaders.DatasetNotDownloaded, match="kaggle"):
        loaders.load_paysim(
            tmp_path / "absent.csv",
            source={
                "url": "https://www.kaggle.com/datasets/ealaxi/paysim1",
                "place_in": "data/raw/",
            },
        )


def test_leak_columns_are_never_read_into_the_contract():
    """PaySim's leak columns cannot reach a model because the loader does not read them at all."""
    cfg = yaml.safe_load((CONFIG_DIR / "paysim.yaml").read_text())
    leak = cfg["leakage"]
    forbidden = set(leak["drop_columns"]) | set(leak["balance_columns"])
    assert forbidden & set(loaders.PAYSIM_COLUMNS) == set()


# ── the real files, when they are there ─────────────────────────────────────────
@pytest.mark.parametrize("cfg", ANCHORS, ids=ANCHOR_IDS)
def test_real_rows_reach_the_contract_with_no_provenance(cfg):
    if not anchor_files_present(cfg):
        pytest.skip(f"{cfg['name']} not downloaded")

    rows = loaders.load_from_config({**cfg, "limit": 20_000})
    assert rows, "the loader returned nothing"
    for t in rows:
        assert t.vector_id is None and t.attack_run_id is None
        assert t.amount > 0
        assert isinstance(t.ts, datetime)
        assert t.rail in tuple(Rail)


@pytest.mark.parametrize("cfg", ANCHORS, ids=ANCHOR_IDS)
def test_the_committed_boundary_partitions_the_real_anchor_both_ways(cfg):
    """Not a tautology: a boundary derived from the full file must still split the sample."""
    if not anchor_files_present(cfg):
        pytest.skip(f"{cfg['name']} not downloaded")

    split = CommittedSplit.load(cfg["name"], SPLIT_DIR)
    rows = loaders.load_from_config({**cfg, "sample": {"sample_fraction": 0.02}})
    train, test = split.apply(rows)
    assert train and test, "the committed boundary emptied one side of the split"
    assert max(t.ts for t in train) < min(t.ts for t in test)
    assert any(t.is_fraud for t in train) and any(t.is_fraud for t in test)


@pytest.mark.parametrize("cfg", ANCHORS, ids=ANCHOR_IDS)
def test_rerunning_the_loader_reproduces_the_identical_partition(cfg):
    """The ticket's actual promise: someone re-running next week gets today's split.

    Both halves have to hold — the entity sample must select the same rows, and the committed
    boundary must cut them in the same place — so this compares txn_id sets, not counts.
    """
    if not anchor_files_present(cfg):
        pytest.skip(f"{cfg['name']} not downloaded")

    split = CommittedSplit.load(cfg["name"], SPLIT_DIR)
    small = {**cfg, "sample": {"sample_fraction": 0.02}}

    first_train, first_test = split.apply(loaders.load_from_config(small))
    second_train, second_test = split.apply(loaders.load_from_config(small))

    assert [t.txn_id for t in first_train] == [t.txn_id for t in second_train]
    assert [t.txn_id for t in first_test] == [t.txn_id for t in second_test]


def test_amlsim_typologies_stay_off_the_contract():
    """The typology is a label-side annotation. On a real row, `vector_id` would be a label path."""
    cfg = next((c for c in ANCHORS if c["name"] == "amlsim"), None)
    if not cfg or not anchor_files_present(cfg):
        pytest.skip("amlsim not downloaded")

    typologies = loaders.amlsim_typologies()
    assert set(typologies.values()) == set(cfg["fraud"]["typologies_present"])
    # one entry per fraud row, and every key is a contract txn_id rather than a raw TX_ID
    assert len(typologies) == CommittedSplit.load("amlsim", SPLIT_DIR).stats["full"]["fraud"]
    assert all(key.startswith("amlsim-") for key in typologies)

    rows = loaders.load_from_config({**cfg, "limit": 5_000})
    assert all(t.vector_id is None for t in rows)
