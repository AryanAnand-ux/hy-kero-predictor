import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import get_baseline_models, select_best_model


def test_elasticnet_uses_stronger_regularization():
    model = get_baseline_models()["ElasticNet"]

    assert model.alpha == 0.05
    assert model.l1_ratio == 0.5
    assert model.max_iter == 20000


def test_select_best_model_prefers_lower_cv_rmse():
    all_results = {
        "ElasticNet": {"test": {"rmse": 2.50}},
        "Lasso Tuned": {"test": {"rmse": 2.60}},
    }
    cv_results = {
        "ElasticNet": {"mean_rmse": 3.90},
        "Lasso Tuned": {"mean_rmse": 3.30},
    }

    assert select_best_model(all_results, cv_results) == "Lasso Tuned"
