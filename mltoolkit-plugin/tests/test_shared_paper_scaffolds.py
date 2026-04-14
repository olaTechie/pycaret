"""methods_md / model_card / checklists — paper-reporting scaffolds."""
from references._shared.methods_md import render_methods
from references._shared.model_card import render_model_card
from references._shared import checklists


def test_methods_renders_all_nine_sections_even_with_empty_context():
    md = render_methods({})
    for heading in ("## 1. Data source", "## 2. Preprocessing",
                    "## 3. Split strategy", "## 4. Model search space",
                    "## 5. Tuning procedure", "## 6. Performance metrics",
                    "## 7. Fairness analysis", "## 8. Calibration",
                    "## 9. Software"):
        assert heading in md


def test_methods_includes_holdout_metrics():
    md = render_methods({"holdout_metrics": {"roc_auc": 0.85, "f1": 0.72}})
    assert "roc_auc: 0.85" in md
    assert "f1: 0.72" in md


def test_model_card_has_required_sections():
    md = render_model_card({})
    for heading in ("## Model details", "## Intended use", "## Training data",
                    "## Evaluation data", "## Performance", "## Fairness",
                    "## Ethical considerations", "## Caveats and recommendations",
                    "## Contact"):
        assert heading in md


def test_tripod_checklist_has_table():
    md = checklists.tripod_ai_checklist()
    assert "| # | Item | Reported? | Where |" in md
    assert "TRIPOD+AI checklist" in md


def test_stard_and_consort_checklists_render():
    stard = checklists.stard_checklist()
    consort = checklists.consort_ai_checklist()
    assert "STARD" in stard
    assert "CONSORT-AI" in consort
