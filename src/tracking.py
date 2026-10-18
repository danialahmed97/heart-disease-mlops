"""MLflow logging helpers: plots, reports and model registration."""
import logging
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
from mlflow.client import MlflowClient  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay,  # noqa: E402
                             PrecisionRecallDisplay, RocCurveDisplay,
                             classification_report)

from src.config import EXPERIMENT_NAME, MODEL_NAME, ROOT  # noqa: E402
from src.features import transformed_feature_names  # noqa: E402

logger = logging.getLogger(__name__)


def init_mlflow() -> None:
    """Point MLflow at the local ./mlruns store and select the experiment."""
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


def feature_importance(pipeline) -> tuple[list[str], np.ndarray] | None:
    """Return (names, importances) for the fitted estimator, if available."""
    model = pipeline.named_steps["model"]
    names = transformed_feature_names(pipeline)
    if hasattr(model, "feature_importances_"):
        return names, np.asarray(model.feature_importances_)
    if hasattr(model, "coef_"):
        return names, np.abs(model.coef_).ravel()
    return None


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def make_plots(res: dict[str, Any], y_test, out: Path) -> list[Path]:
    """Create confusion matrix, ROC, PR and importance PNGs."""
    paths = []
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, res["y_pred"], ax=ax, cmap="Blues",
        display_labels=["Healthy", "Disease"])
    ax.set_title(f"{res['name']} - confusion matrix")
    paths.append(_save(fig, out / "confusion_matrix.png"))

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, res["y_proba"], ax=ax,
                                     name=res["name"])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC curve")
    paths.append(_save(fig, out / "roc_curve.png"))

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_test, res["y_proba"], ax=ax,
                                            name=res["name"])
    ax.set_title("Precision-recall curve")
    paths.append(_save(fig, out / "precision_recall_curve.png"))

    imp = feature_importance(res["pipeline"])
    if imp is not None:
        names, vals = imp
        order = np.argsort(vals)[-15:]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(np.array(names)[order], vals[order], color="steelblue")
        ax.set_title(f"{res['name']} - top feature importances")
        paths.append(_save(fig, out / "feature_importance.png"))
    return paths


def log_run(res: dict[str, Any], X_train, X_test, y_test) -> str:
    """Log one model's params, metrics, artifacts and model; return run id."""
    with mlflow.start_run(run_name=res["name"]) as run:
        mlflow.set_tag("model_family", res["name"])
        mlflow.log_params({k.replace("model__", ""): v
                           for k, v in res["best_params"].items()})
        mlflow.log_metric("cv_roc_auc_mean", res["cv_roc_auc_mean"])
        mlflow.log_metric("cv_roc_auc_std", res["cv_roc_auc_std"])
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            mlflow.log_metric(f"test_{key}", res[f"test_{key}"])
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for p in make_plots(res, y_test, tmp_path):
                mlflow.log_artifact(str(p), artifact_path="plots")
            report = classification_report(
                y_test, res["y_pred"], target_names=["Healthy", "Disease"])
            (tmp_path / "classification_report.txt").write_text(report)
            mlflow.log_artifact(str(tmp_path / "classification_report.txt"),
                                artifact_path="reports")
        signature = infer_signature(X_test, res["y_proba"])
        mlflow.sklearn.log_model(
            res["pipeline"], artifact_path="model", signature=signature,
            input_example=X_train.head(3))
        logger.info("Logged run %s (%s)", run.info.run_id, res["name"])
        return run.info.run_id


def tag_and_register_best(run_id: str, name: str) -> str:
    """Tag the best run and register its model in the MLflow registry."""
    client = MlflowClient()
    client.set_tag(run_id, "best_model", "true")
    mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    client.set_model_version_tag(MODEL_NAME, mv.version, "model_family", name)
    logger.info("Registered %s v%s (alias champion)", MODEL_NAME, mv.version)
    return str(mv.version)
