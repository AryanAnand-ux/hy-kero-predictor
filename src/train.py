

import pandas as pd
import numpy as np
import json
import warnings
import joblib
from pathlib import Path

from sklearn.linear_model   import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, BayesianRidge
from sklearn.ensemble       import RandomForestRegressor
from sklearn.svm            import SVR
from sklearn.base           import clone
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost  as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

BASE   = Path(__file__).resolve().parent.parent
PROC   = BASE / "data" / "processed"
MODELS = BASE / "data" / "models"


# ── Metrics helper ────────────────────────────────────────────────────────────
def metrics(y_true, y_pred, label="") -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true.clip(lower=1e-6))) * 100)
    if label:
        print(f"  {label:<30} RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.4f}  MAPE={mape:.2f}%")
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


# ── Model definitions ─────────────────────────────────────────────────────────
def get_baseline_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge":             Ridge(alpha=1.0),
        "Lasso":             Lasso(alpha=0.02, max_iter=20000),
        "ElasticNet":        ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=20000),
        "Huber":             HuberRegressor(alpha=0.001, epsilon=1.35, max_iter=20000),
        "BayesianRidge":     BayesianRidge(max_iter=1000),
        "SVR":               SVR(kernel="rbf", C=1.0, epsilon=0.2),
        "Random Forest":     RandomForestRegressor(
                                 n_estimators=100, max_depth=3,
                                 min_samples_leaf=15, min_samples_split=20,
                                 random_state=42, n_jobs=-1),
        "XGBoost":           xgb.XGBRegressor(
                                 n_estimators=100, learning_rate=0.03,
                                 max_depth=2, subsample=0.7,
                                 min_child_weight=15, reg_alpha=1.0, reg_lambda=5.0,
                                 random_state=42, n_jobs=-1),
        "LightGBM":          lgb.LGBMRegressor(
                                 n_estimators=100, learning_rate=0.03,
                                 max_depth=2, num_leaves=4, subsample=0.7,
                                 min_child_samples=20, reg_alpha=1.0, reg_lambda=5.0,
                                 random_state=42, n_jobs=-1, verbose=-1),
    }


# ── Walk-forward cross-validation (time-series aware) ────────────────────────
def cv_score(model, X, y, n_splits=5) -> tuple[float, list]:
    """Returns (mean_rmse, fold_rmse_list) using TimeSeriesSplit.
    
    gap=2 prevents lag feature leakage: lag-1/lag-2 features from the
    last 2 training rows would otherwise predict the first validation rows.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=2)
    scores = []
    for tr, va in tscv.split(X):
        m = clone(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[va])
        scores.append(np.sqrt(mean_squared_error(y.iloc[va], pred)))
    return float(np.mean(scores)), scores


# ── Optuna fine-tuning ────────────────────────────────────────────────────────


def tune_lasso(X_train, y_train, n_trials=60) -> dict:
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
        model = Lasso(alpha=alpha, max_iter=20000)
        mean_rmse, _ = cv_score(model, X_train, y_train)
        return mean_rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_elasticnet(X_train, y_train, n_trials=60) -> dict:
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 2.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.05, 0.95)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000)
        mean_rmse, _ = cv_score(model, X_train, y_train)
        return mean_rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_ridge(X_train, y_train, n_trials=60) -> dict:
    """Tune Ridge alpha — Ridge is competitive but was previously untuned."""
    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.01, 100.0, log=True)
        model = Ridge(alpha=alpha)
        mean_rmse, _ = cv_score(model, X_train, y_train)
        return mean_rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_huber(X_train, y_train, n_trials=60) -> dict:
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
        epsilon = trial.suggest_float("epsilon", 1.05, 3.0)
        model = HuberRegressor(alpha=alpha, epsilon=epsilon, max_iter=20000)
        mean_rmse, _ = cv_score(model, X_train, y_train)
        return mean_rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def select_best_model(all_results: dict, cv_results: dict) -> str:
    """
    Choose the strongest regularized model based strictly on CV RMSE to avoid data leakage
    from the held-out test set. Includes Ensemble and regularized linear models.
    """
    candidate_names = [
        name for name in cv_results
        if name in all_results and not any(token in name for token in ["XGBoost", "LightGBM", "Random Forest", "SVR"])
    ]
    # Allow Ensemble and regularized/Bayesian linear models
    eligible_names = [
        name for name in candidate_names
        if name == "Ensemble" or any(token in name for token in [
            "ElasticNet", "Lasso", "Huber", "Ridge", "Linear Regression", "BayesianRidge"
        ])
    ]
    if eligible_names:
        candidate_names = eligible_names

    if not candidate_names:
        # Fallback to the lowest CV RMSE among all available models.
        candidate_names = [name for name in cv_results if name in all_results]

    return min(candidate_names, key=lambda name: cv_results[name]["mean_rmse"])


# ── Simple Ensemble ──────────────────────────────────────────────────────────
class WeightedEnsemble:
    """Weighted average ensemble of multiple models."""

    def __init__(self, models: list, weights: list | None = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

    def predict(self, X) -> np.ndarray:
        preds = np.array([m.predict(X) for m in self.models])
        return np.average(preds, axis=0, weights=self.weights)

    def get_params(self, deep=False):
        return {"models": self.models, "weights": self.weights}


# ── Main training pipeline ────────────────────────────────────────────────────
def run():
    print("\n=== Phase 3 & 4: Model Training + Fine-Tuning ===\n")

    X_train = pd.read_csv(PROC / "X_train.csv")
    X_test  = pd.read_csv(PROC / "X_test.csv")
    y_train = pd.read_csv(PROC / "y_train.csv").squeeze()
    y_test  = pd.read_csv(PROC / "y_test.csv").squeeze()

    print(f"  Train: {X_train.shape} | Test: {X_test.shape}\n")

    # ── Step 1: Train all baseline models ──
    print("--- Baseline Models ---")
    all_results = {}
    cv_results  = {}
    trained_models = {}
    for name, model in get_baseline_models().items():
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test  = model.predict(X_test)
        train_m = metrics(y_train, pred_train, f"[Train] {name}")
        test_m  = metrics(y_test,  pred_test,  f"[Test]  {name}")
        all_results[name] = {"train": train_m, "test": test_m}
        trained_models[name] = model
        joblib.dump(model, MODELS / f"{name.replace(' ', '_')}.pkl", compress=3)

        # Walk-forward CV score
        mean_cv, fold_scores = cv_score(model, X_train, y_train)
        cv_results[name] = {"mean_rmse": round(mean_cv, 3),
                            "fold_scores": [round(s, 3) for s in fold_scores]}
        print(f"    CV RMSE: {mean_cv:.3f} (folds: {[f'{s:.2f}' for s in fold_scores]})")
        print()

    # ── Step 2: Fine-tune linear regularized models ──
    print("--- Fine-Tuning Linear Regularized Models ---")
    
    # Lasso
    best_lasso_params = tune_lasso(X_train, y_train, n_trials=60)
    print(f"  Best Lasso params: {best_lasso_params}")
    lasso_tuned = Lasso(**best_lasso_params, max_iter=20000)
    lasso_tuned.fit(X_train, y_train)
    lasso_train_m = metrics(y_train, lasso_tuned.predict(X_train), "[Train] Lasso Tuned")
    lasso_test_m  = metrics(y_test,  lasso_tuned.predict(X_test),  "[Test]  Lasso Tuned")
    all_results["Lasso Tuned"] = {"train": lasso_train_m, "test": lasso_test_m}
    trained_models["Lasso Tuned"] = lasso_tuned
    joblib.dump(lasso_tuned, MODELS / "Lasso_Tuned.pkl", compress=3)
    mean_cv, fold_scores = cv_score(lasso_tuned, X_train, y_train)
    cv_results["Lasso Tuned"] = {"mean_rmse": round(mean_cv, 3),
                                  "fold_scores": [round(s, 3) for s in fold_scores]}
    print(f"    CV RMSE: {mean_cv:.3f}\n")

    # ElasticNet
    best_enet_params = tune_elasticnet(X_train, y_train, n_trials=60)
    print(f"  Best ElasticNet params: {best_enet_params}")
    enet_tuned = ElasticNet(**best_enet_params, max_iter=20000)
    enet_tuned.fit(X_train, y_train)
    enet_train_m = metrics(y_train, enet_tuned.predict(X_train), "[Train] ElasticNet Tuned")
    enet_test_m  = metrics(y_test,  enet_tuned.predict(X_test),  "[Test]  ElasticNet Tuned")
    all_results["ElasticNet Tuned"] = {"train": enet_train_m, "test": enet_test_m}
    trained_models["ElasticNet Tuned"] = enet_tuned
    joblib.dump(enet_tuned, MODELS / "ElasticNet_Tuned.pkl", compress=3)
    mean_cv, fold_scores = cv_score(enet_tuned, X_train, y_train)
    cv_results["ElasticNet Tuned"] = {"mean_rmse": round(mean_cv, 3),
                                       "fold_scores": [round(s, 3) for s in fold_scores]}
    print(f"    CV RMSE: {mean_cv:.3f}\n")

    # Ridge
    best_ridge_params = tune_ridge(X_train, y_train, n_trials=60)
    print(f"  Best Ridge params: {best_ridge_params}")
    ridge_tuned = Ridge(**best_ridge_params)
    ridge_tuned.fit(X_train, y_train)
    ridge_train_m = metrics(y_train, ridge_tuned.predict(X_train), "[Train] Ridge Tuned")
    ridge_test_m  = metrics(y_test,  ridge_tuned.predict(X_test),  "[Test]  Ridge Tuned")
    all_results["Ridge Tuned"] = {"train": ridge_train_m, "test": ridge_test_m}
    trained_models["Ridge Tuned"] = ridge_tuned
    joblib.dump(ridge_tuned, MODELS / "Ridge_Tuned.pkl", compress=3)
    mean_cv, fold_scores = cv_score(ridge_tuned, X_train, y_train)
    cv_results["Ridge Tuned"] = {"mean_rmse": round(mean_cv, 3),
                                  "fold_scores": [round(s, 3) for s in fold_scores]}
    print(f"    CV RMSE: {mean_cv:.3f}\n")

    # Huber
    best_huber_params = tune_huber(X_train, y_train, n_trials=60)
    print(f"  Best Huber params: {best_huber_params}")
    huber_tuned = HuberRegressor(**best_huber_params, max_iter=20000)
    huber_tuned.fit(X_train, y_train)
    huber_train_m = metrics(y_train, huber_tuned.predict(X_train), "[Train] Huber Tuned")
    huber_test_m  = metrics(y_test,  huber_tuned.predict(X_test),  "[Test]  Huber Tuned")
    all_results["Huber Tuned"] = {"train": huber_train_m, "test": huber_test_m}
    trained_models["Huber Tuned"] = huber_tuned
    joblib.dump(huber_tuned, MODELS / "Huber_Tuned.pkl", compress=3)
    mean_cv, fold_scores = cv_score(huber_tuned, X_train, y_train)
    cv_results["Huber Tuned"] = {"mean_rmse": round(mean_cv, 3),
                                  "fold_scores": [round(s, 3) for s in fold_scores]}
    print(f"    CV RMSE: {mean_cv:.3f}\n")

    # ── Step 3: Build weighted ensemble of top 3 models by CV RMSE (leak-free) ──
    print("--- Building Ensemble ---")
    cv_candidates = {name: res["mean_rmse"] for name, res in cv_results.items() if name in trained_models and name != "Linear Regression"}
    sorted_models = sorted(cv_candidates.items(), key=lambda kv: kv[1])
    top3_names = [name for name, _ in sorted_models[:3]]
    top3_models = [trained_models[n] for n in top3_names]

    # Inverse-RMSE weighting based on CV scores
    top3_rmses = [cv_candidates[n] for n in top3_names]
    inv_rmses = [1.0 / max(r, 1e-6) for r in top3_rmses]
    total_inv = sum(inv_rmses)
    weights = [w / total_inv for w in inv_rmses]

    ensemble = WeightedEnsemble(top3_models, weights)
    ens_pred_train = ensemble.predict(X_train)
    ens_pred_test  = ensemble.predict(X_test)
    ens_train_m = metrics(y_train, ens_pred_train, "[Train] Ensemble")
    ens_test_m  = metrics(y_test,  ens_pred_test,  "[Test]  Ensemble")
    all_results["Ensemble"] = {"train": ens_train_m, "test": ens_test_m}
    trained_models["Ensemble"] = ensemble
    print(f"  Ensemble components: {top3_names}")
    print(f"  Weights: {[f'{w:.3f}' for w in weights]}")
    joblib.dump(ensemble, MODELS / "Ensemble.pkl", compress=3)
    
    # Calculate Ensemble CV score chronologically without leakage
    def cv_score_ensemble(top3_names, X, y, n_splits=5) -> float:
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=2)
        scores = []
        for tr, va in tscv.split(X):
            fold_preds = []
            fold_rmses = []
            for name in top3_names:
                base_model = trained_models[name]
                m = clone(base_model)
                m.fit(X.iloc[tr], y.iloc[tr])
                pred_va = m.predict(X.iloc[va])
                fold_preds.append(pred_va)
                # training score of fold for weighting
                pred_tr = m.predict(X.iloc[tr])
                fold_rmses.append(np.sqrt(mean_squared_error(y.iloc[tr], pred_tr)))
            
            inv = [1.0 / max(r, 1e-6) for r in fold_rmses]
            t_inv = sum(inv)
            w_fold = [w / t_inv for w in inv]
            avg_pred = np.average(fold_preds, axis=0, weights=w_fold)
            scores.append(np.sqrt(mean_squared_error(y.iloc[va], avg_pred)))
        return float(np.mean(scores))

    ens_cv_rmse = cv_score_ensemble(top3_names, X_train, y_train)
    cv_results["Ensemble"] = {"mean_rmse": round(ens_cv_rmse, 3), "fold_scores": []}
    print(f"    CV RMSE: {ens_cv_rmse:.3f}\n")

    # ── Step 4: Pick best model by held-out RMSE among regularized models ──
    best_name = select_best_model(all_results, cv_results)
    best_model = trained_models[best_name]
    print(f"\n  >>> Best model (selected by CV RMSE): {best_name} (CV RMSE = {cv_results[best_name]['mean_rmse']:.3f}, Test RMSE = {all_results[best_name]['test']['rmse']:.3f})")
    joblib.dump(best_model, MODELS / "best_model.pkl", compress=3)
    joblib.dump(best_name,  MODELS / "best_model_name.pkl", compress=3)

    # ── Step 5: Save feature importances if available ──
    feature_cols = joblib.load(MODELS / "feature_cols.pkl")
    importances = None
    if isinstance(best_model, WeightedEnsemble):
        # Aggregate feature importances from components
        component_importances = []
        valid_weights = []
        for model, weight in zip(best_model.models, best_model.weights):
            imp = None
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
            elif hasattr(model, "coef_"):
                imp = np.abs(model.coef_)
            if imp is not None:
                component_importances.append(imp)
                valid_weights.append(weight)
        if component_importances:
            w_sum = sum(valid_weights)
            norm_weights = [w / w_sum for w in valid_weights]
            importances = np.average(component_importances, axis=0, weights=norm_weights)
    else:
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
        elif hasattr(best_model, "coef_"):
            importances = np.abs(best_model.coef_)
            
    if importances is not None:
        fi = pd.DataFrame({
            "feature":    feature_cols,
            "importance": importances
        }).sort_values("importance", ascending=False)
        fi.to_csv(PROC / "feature_importance.csv", index=False)
        print(f"\n  Top 10 features:")
        print(fi.head(10).to_string(index=False))

    # ── Step 6: Save full predictions for history endpoint ──
    scaler = joblib.load(MODELS / "scaler.pkl")
    X_full = pd.read_csv(PROC / "X_full.csv")
    X_full_sc = pd.DataFrame(scaler.transform(X_full), columns=feature_cols)
    y_full = pd.read_csv(PROC / "y_full.csv").squeeze()

    preds = best_model.predict(X_full_sc)

    merged = pd.read_csv(PROC / "merged_dataset.csv", parse_dates=["sample_ts"])
    merged = merged.iloc[len(merged) - len(preds):].reset_index(drop=True)
    history_df = pd.DataFrame({
        "sample_ts":  merged["sample_ts"].values,
        "shift":      merged["shift"].values,
        "actual":     y_full.values,
        "predicted":  preds,
    })
    history_df["residual"] = history_df["actual"] - history_df["predicted"]
    history_df.to_csv(PROC / "predictions_history.csv", index=False)
    print(f"\n  Saved predictions_history.csv ({len(history_df)} rows)")

    # ── Step 7: Save metrics JSON + CV results for API ──
    with open(PROC / "model_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(PROC / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)

    print(f"  Saved model_metrics.json + cv_results.json")
    print("\n=== Training Complete ===")
    return all_results, best_name


if __name__ == "__main__":
    run()
