"""OR (clause) languages: parameters, names and clause accessors."""

from pathlib import Path

import pytest

# Sibling checkout layout (the Cargo path dependency relies on it as well)
SAMPLES = Path(__file__).resolve().parents[2] / "gpredomics" / "samples" / "Qin2014"


def test_or_language_parameters_are_accepted():
    from gpredomicspy import Param

    p = Param()
    p.set_string("algo", "beam")
    p.set_string("language", "bin-or,ter-or,ratio-or")
    p.set("max_clause_size", 2)


def _beam_config(tmp_path: Path, language: str) -> Path:
    cfg = tmp_path / "param.yaml"
    cfg.write_text(
        f"""
general:
  seed: 42
  algo: beam
  language: {language}
  data_type: prev
  thread_number: 2
  k_penalty: 0.0001
  keep_trace: true
  log_level: warning
data:
  X: "{SAMPLES / 'Xtrain.tsv'}"
  y: "{SAMPLES / 'Ytrain.tsv'}"
  features_in_rows: true
  holdout_ratio: 0.0
  feature_minimal_prevalence_pct: 10
  classes: ["healthy", "cirrhosis", "unknown"]
beam:
  method: LimitedExhaustive
  k_start: 1
  k_stop: 4
  best_models_criterion: 10
  max_nb_of_models: 2000
  max_clause_size: 2
"""
    )
    return cfg


@pytest.mark.skipif(not SAMPLES.is_dir(), reason="gpredomics samples not available")
def test_beam_ter_or_exposes_clauses(tmp_path):
    from gpredomicspy import Param, fit

    p = Param()
    p.load(str(_beam_config(tmp_path, "ter-or")))
    exp = fit(p)
    pop = exp.best_population()
    assert len(pop) > 0

    seen_or = False
    seen_grouped = False
    for i in range(len(pop)):
        ind = pop.get_individual(i)
        metrics = ind.get_metrics()
        features = ind.get_features()
        clauses = ind.get_clauses()
        assert metrics["language"] in ("ternary-or", "binary-or"), metrics["language"]
        assert clauses is not None
        seen_or = True
        # clauses partition the features and share one coefficient each
        flat = sorted(f for clause in clauses for f in clause)
        assert flat == sorted(features)
        for clause in clauses:
            assert 1 <= len(clause) <= 2
            assert len({features[f] for f in clause}) == 1
            if len(clause) > 1:
                seen_grouped = True
    assert seen_or
    assert seen_grouped, "expected at least one clause with two features in the best population"


@pytest.mark.skipif(not SAMPLES.is_dir(), reason="gpredomics samples not available")
def test_classic_models_have_no_clauses(tmp_path):
    from gpredomicspy import Param, fit

    p = Param()
    p.load(str(_beam_config(tmp_path, "ter")))
    exp = fit(p)
    pop = exp.best_population()
    assert len(pop) > 0
    ind = pop.get_individual(0)
    assert ind.get_metrics()["language"] == "ternary"
    assert ind.get_clauses() is None
