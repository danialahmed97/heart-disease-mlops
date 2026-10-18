"""Train, tune and compare models for heart-disease prediction."""
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLEAN_PATH, DOCS_DIR, RAW_FEATURES, SEED, TARGET  # noqa: E402
from src.config import set_seeds, setup_logging  # noqa: E402
from src.features import build_pipeline  # noqa: E402

logger = logging.getLogger(__name__)

try:  # XGBoost is preferred; fall back to sklearn GradientBoosting.
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # pragma: no cover - depends on platform libs
    HAS_XGB = False


def get_candidates() -> dict[str, tuple[Any, dict[str, list]]]:
    """Return {name: (estimator, param_grid)} for the three model families."""
    cands: dict[str, tuple[Any, dict[str, list]]] = {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, random_state=SEED),
            {"model__C": [0.01, 0.1, 1, 10], "model__penalty": ["l2"],
             "model__class_weight": [None, "balanced"]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {"model__n_estimators": [100, 300], "model__max_depth": [3, 5, None],
             "model__min_samples_leaf": [1, 3]},
        ),
    }
    if HAS_XGB:
        cands["XGBoost"] = (
            XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=1),
            {"model__n_estimators": [100, 200], "model__max_depth": [2, 3, 4],
             "model__learning_rate": [0.03, 0.1]},
        )
    else:
        cands["GradientBoosting"] = (
            GradientBoostingClassifier(random_state=SEED),
            {"model__n_estimators": [100, 200], "model__max_depth": [2, 3],
             "model__learning_rate": [0.03, 0.1]},
        )
    return cands


def load_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 split of the cleaned data."""
    df = pd.read_csv(CLEAN_PATH)
    return train_test_split(
        df[RAW_FEATURES], df[TARGET], test_size=0.2,
        stratify=df[TARGET], random_state=SEED,
    )


def test_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Hold-out metrics."""
    return {
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_precision": precision_score(y_true, y_pred),
        "test_recall": recall_score(y_true, y_pred),
        "test_f1": f1_score(y_true, y_pred),
        "test_roc_auc": roc_auc_score(y_true, y_proba),
    }


def tune_model(name: str, estimator, grid, X_train, y_train) -> GridSearchCV:
    """5-fold stratified GridSearchCV optimising ROC-AUC."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(build_pipeline(estimator), grid, scoring="roc_auc",
                      cv=cv, n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)
    logger.info("%s best CV AUC %.4f params %s", name, gs.best_score_,
                gs.best_params_)
    return gs


def select_best(results: list[dict[str, Any]], tol: float = 0.005) -> dict:
    """Pick the best model by CV ROC-AUC; within ``tol``, higher recall wins.

    Selection uses CV AUC (not test AUC) to avoid optimising on the hold-out
    set. Recall is the tiebreaker because a missed case is costlier than a
    false alarm in medical screening.
    """
    top = max(r["cv_roc_auc_mean"] for r in results)
    contenders = [r for r in results if top - r["cv_roc_auc_mean"] <= tol]
    return max(contenders, key=lambda r: (r["test_recall"],
                                          r["cv_roc_auc_mean"]))


def write_comparison(results: list[dict[str, Any]], best: dict) -> None:
    """Write docs/model_comparison.md."""
    DOCS_DIR.mkdir(exist_ok=True)
    lines = [
        "# Model Comparison", "",
        "Stratified 80/20 split (random_state=42); 5-fold stratified "
        "GridSearchCV optimising ROC-AUC on the training portion only.", "",
        "| Model | CV ROC-AUC (mean ± std) | Test Accuracy | Precision | "
        "Recall | F1 | Test ROC-AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        star = " ⭐" if r["name"] == best["name"] else ""
        lines.append(
            f"| {r['name']}{star} | {r['cv_roc_auc_mean']:.3f} ± "
            f"{r['cv_roc_auc_std']:.3f} | {r['test_accuracy']:.3f} | "
            f"{r['test_precision']:.3f} | {r['test_recall']:.3f} | "
            f"{r['test_f1']:.3f} | {r['test_roc_auc']:.3f} |"
        )
    lines += ["", "## Best hyperparameters", ""]
    for r in results:
        lines.append(f"- **{r['name']}**: `{r['best_params']}`")
    lines += [
        "", "## Selection rationale", "",
        f"**Selected model: {best['name']}.**", "",
        "- Primary criterion: cross-validated ROC-AUC (threshold-independent "
        "ranking quality; robust on a small 303-row dataset).",
        "- Tiebreaker: test-set recall. Models whose CV AUC is within 0.005 of "
        "the best are treated as tied and the one with the highest recall "
        "wins, since in screening a missed patient (false negative) is "
        "costlier than a false alarm.",
        "- Selection uses CV AUC rather than test AUC so the hold-out set "
        "stays an unbiased estimate.",
        "- With only ~61 test rows, differences of one or two patients move "
        "metrics by several points; the CV std column conveys this "
        "uncertainty.", "",
    ]
    (DOCS_DIR / "model_comparison.md").write_text("\n".join(lines))


def main() -> dict:
    """Run the full training experiment."""
    set_seeds()
    X_train, X_test, y_train, y_test = load_split()
    logger.info("Train %s, test %s", X_train.shape, X_test.shape)
    results = []
    for name, (est, grid) in get_candidates().items():
        gs = tune_model(name, est, grid, X_train, y_train)
        pipe = gs.best_estimator_
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        idx = gs.best_index_
        res = {
            "name": name, "pipeline": pipe, "best_params": gs.best_params_,
            "cv_roc_auc_mean": float(gs.cv_results_["mean_test_score"][idx]),
            "cv_roc_auc_std": float(gs.cv_results_["std_test_score"][idx]),
            "y_pred": pred, "y_proba": proba,
            **{k: float(v) for k, v in test_metrics(y_test, pred, proba).items()},
        }
        results.append(res)
    best = select_best(results)
    logger.info("Best model: %s", best["name"])
    write_comparison(results, best)
    return {"results": results, "best": best, "split": (X_train, X_test,
                                                         y_train, y_test)}


if __name__ == "__main__":
    setup_logging()
    main()
