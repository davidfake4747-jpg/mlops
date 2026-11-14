"""Сравнение двух фиксированных моделей, MLflow и выбранный артефакт API."""

import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from service.predict import FEATURES, feature_frame, save_model


def build_model(name):
    if name == "baseline":
        return DummyRegressor(strategy="median")
    if name == "linear":
        return make_pipeline(StandardScaler(), LinearRegression())
    raise ValueError(f"Неизвестная модель: {name}")


def metrics(y, predictions):
    predictions = np.asarray(predictions, dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("Неконечные предсказания")
    return {
        "mae_rub": float(mean_absolute_error(y, predictions)),
        "mape_pct": float(100 * mean_absolute_percentage_error(y, predictions)),
    }


def validate_split(train, test, n_splits):
    for frame in (train, test):
        if frame.empty or frame["id"].isna().any() or frame["id"].duplicated().any():
            raise ValueError("Выборки должны содержать уникальные непустые ID")
        if frame["address"].isna().any() or frame["address"].astype(str).str.strip().eq("").any():
            raise ValueError("Не у всех объектов указан адрес")
        values = frame[[*FEATURES, "price"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (frame["price"] <= 0).any():
            raise ValueError("Некорректные признаки или целевая цена")
    if set(train["id"]) & set(test["id"]) or set(train["address"]) & set(test["address"]):
        raise ValueError("Утечка данных между train и holdout")
    if set(train["cv_fold"]) != set(range(n_splits)):
        raise ValueError("Некорректные validation-folds")
    if not train.groupby("address")["cv_fold"].nunique().eq(1).all():
        raise ValueError("Один адрес попал в разные validation-folds")


def evaluate(name, train, test, n_splits):
    fold_metrics = []
    valid_predictions = True
    for fold in range(n_splits):
        fit = train.loc[train["cv_fold"].ne(fold)]
        val = train.loc[train["cv_fold"].eq(fold)]
        model = build_model(name).fit(feature_frame(fit), fit["price"])
        predictions = model.predict(feature_frame(val))
        valid_predictions &= bool(np.isfinite(predictions).all() and (predictions > 0).all())
        fold_metrics.append({"fold": fold, **metrics(val["price"], predictions)})
    model = build_model(name).fit(feature_frame(train), train["price"])
    predictions = model.predict(feature_frame(test))
    train_predictions = model.predict(feature_frame(train))
    valid_predictions &= bool(np.isfinite(predictions).all() and (predictions > 0).all())
    valid_predictions &= bool(np.isfinite(train_predictions).all() and (train_predictions > 0).all())
    result = {
        "cv_mae_mean_rub": float(np.mean([m["mae_rub"] for m in fold_metrics])),
        "cv_mae_std_rub": float(np.std([m["mae_rub"] for m in fold_metrics])),
        "cv_mape_mean_pct": float(np.mean([m["mape_pct"] for m in fold_metrics])),
        "cv_mape_std_pct": float(np.std([m["mape_pct"] for m in fold_metrics])),
        "train_mae_rub": metrics(train["price"], train_predictions)["mae_rub"],
        "train_mape_pct": metrics(train["price"], train_predictions)["mape_pct"],
        "holdout_mae_rub": metrics(test["price"], predictions)["mae_rub"],
        "holdout_mape_pct": metrics(test["price"], predictions)["mape_pct"],
        "positive_finite_predictions": valid_predictions,
        "folds": fold_metrics,
    }
    return model, result, predictions


def select_model(results):
    baseline, linear = results["baseline"], results["linear"]
    return "linear" if (
        linear["positive_finite_predictions"]
        and linear["cv_mae_mean_rub"] < baseline["cv_mae_mean_rub"]
        and linear["holdout_mae_rub"] < baseline["holdout_mae_rub"]
    ) else "baseline"


def git_metadata():
    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()
    return {"git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain"))}


def log_experiments(models, results, selected, train, test, config, data_hash):
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    Path("mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(config["tracking_uri"])
    experiment = mlflow.get_experiment_by_name(config["experiment_name"])
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            config["experiment_name"], artifact_location=Path("mlruns/artifacts").resolve().as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    provenance = {
        **git_metadata(), "data_dvc_md5": data_hash,
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__, "mlflow_version": mlflow.__version__,
        "numpy_version": np.__version__,
    }
    parameters = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    for name, model in models.items():
        result = results[name]
        with mlflow.start_run(experiment_id=experiment_id, run_name=name):
            mlflow.log_params({
                "model": name, "features": ",".join(FEATURES),
                "train_rows": len(train), "holdout_rows": len(test),
                "train_groups": train["address"].nunique(), "holdout_groups": test["address"].nunique(),
                "random_state": parameters["split"]["random_state"],
                "test_size": parameters["split"]["test_size"],
                "n_splits": parameters["split"]["n_splits"],
                **{k: str(v) for k, v in model.get_params(deep=True).items()},
            })
            mlflow.set_tags({**provenance, "selected": name == selected})
            mlflow.log_metrics({k: v for k, v in result.items() if isinstance(v, (float, int))})
            for fold in result["folds"]:
                mlflow.log_metrics({"fold_mae_rub": fold["mae_rub"], "fold_mape_pct": fold["mape_pct"]}, step=fold["fold"])
            example = feature_frame(train).head(3)
            mlflow.sklearn.log_model(
                model, name="model", input_example=example,
                signature=infer_signature(example, model.predict(example)),
                pip_requirements="requirements-model.lock.txt",
            )
            mlflow.log_artifact(f"models/{name}.joblib", artifact_path="api_bundle")
            for path in ("params.yaml", "data.dvc", "requirements-model.lock.txt", "reports/data_quality.json", "reports/split.json", "reports/training.json", "reports/holdout_predictions.csv"):
                mlflow.log_artifact(path, artifact_path="reports")


def main():
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    train = pd.read_csv("features/train.csv", dtype={"id": str})
    test = pd.read_csv("features/test.csv", dtype={"id": str})
    n_splits = params["split"]["n_splits"]
    validate_split(train, test, n_splits)
    data_hash = yaml.safe_load(Path("data.dvc").read_text(encoding="utf-8"))["outs"][0]["md5"]
    models, results = {}, {}
    predictions = test[["id", "price"]].copy()
    for name in ("baseline", "linear"):
        models[name], results[name], predictions[name] = evaluate(name, train, test, n_splits)
        save_model(models[name], name, f"models/{name}.joblib", data_hash)
    selected = select_model(results)
    save_model(models[selected], selected, "models/selected.joblib", data_hash)
    report = {
        "selected_model": selected, "features": list(FEATURES), "data_dvc_md5": data_hash,
        "train_rows": len(train), "holdout_rows": len(test), "models": results,
        "selection_rule": "linear only if positive predictions and lower CV and holdout MAE",
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/training.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    predictions.to_csv("reports/holdout_predictions.csv", index=False)
    log_experiments(models, results, selected, train, test, params["train"], data_hash)
    for name, result in results.items():
        print(f"{name}: CV MAE={result['cv_mae_mean_rub']:.0f} руб.; holdout MAE={result['holdout_mae_rub']:.0f} руб.")
    print(f"Выбрана модель: {selected}")


if __name__ == "__main__":
    main()
