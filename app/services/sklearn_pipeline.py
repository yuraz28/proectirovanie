from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path  # noqa: TC003
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

TaskType = Literal["classification", "regression"]

_MAX_DISTINCT_CLASSES_FOR_CLASSIFICATION = 50
_MIN_ROWS_AFTER_CLEAN = 2
_MIN_CLASSES = 2
_MIN_COLS_FOR_CORRELATION = 2
_MIN_ROWS_FOR_CORRELATION = 2
_FEATURE_IMPORTANCE_TOP = 15
_RF_TREES = 20
_RF_DEPTH = 8
_RF_SEED = 42
_TEST_FRACTION = 0.2


@dataclass(frozen=True, slots=True)
class XYSplit:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def _read_csv(dataset_path: Path) -> pd.DataFrame:
    suffix = dataset_path.suffix.lower()

    if suffix != ".csv":
        msg = "Поддерживаются только CSV-файлы"
        raise ValueError(msg)

    return pd.read_csv(dataset_path, index_col=["Unnamed: 0"])


def _numeric_feature_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    numeric = df.select_dtypes(include=["number"]).columns.tolist()

    return [c for c in numeric if c != target_column]


def _infer_task(y: pd.Series) -> tuple[TaskType, pd.Series]:
    y_num = pd.to_numeric(y, errors="coerce")
    valid = y_num.dropna()

    if valid.empty:
        msg = "В колонке таргета нет числовых значений"
        raise ValueError(msg)

    y_f = valid.astype(float)
    is_whole = np.isclose(y_f, np.round(y_f))
    n_unique = int(valid.nunique())

    if bool(is_whole.all()) and n_unique <= _MAX_DISTINCT_CLASSES_FOR_CLASSIFICATION:
        y_cls = pd.Series(y_num, index=y.index).dropna().astype(int)

        return "classification", y_cls.reindex(y.index)

    y_reg = pd.Series(y_num, index=y.index)

    return "regression", y_reg


def _correlation_matrix_payload(frame: pd.DataFrame) -> dict[str, object]:
    if frame.shape[1] < _MIN_COLS_FOR_CORRELATION:
        msg = "Для корреляции нужны таргет и хотя бы один признак"
        raise ValueError(msg)

    numeric_frame = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

    if numeric_frame.shape[0] < _MIN_ROWS_FOR_CORRELATION:
        columns = [str(c) for c in frame.columns.tolist()]
        n = len(columns)
        eye = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        return {"columns": columns, "values": eye}

    corr = numeric_frame.corr(method="pearson", min_periods=2).fillna(0.0)
    columns = [str(c) for c in corr.columns.tolist()]
    values = [[float(v) for v in row] for row in corr.to_numpy().tolist()]

    payload: dict[str, object] = {"columns": columns, "values": values}

    return payload


def _confusion_matrix_payload(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    labels_sorted = np.sort(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    labels = [str(int(x)) if np.isclose(x, np.round(x)) else str(x) for x in labels_sorted.tolist()]
    matrix = [[int(c) for c in row] for row in cm.tolist()]

    return {"labels": labels, "matrix": matrix}


def _fit_classification(
    *,
    split: XYSplit,
) -> tuple[RandomForestClassifier, dict[str, float], dict[str, object]]:
    clf = RandomForestClassifier(
        n_estimators=_RF_TREES,
        max_depth=_RF_DEPTH,
        random_state=_RF_SEED,
        n_jobs=-1,
    )
    clf.fit(split.x_train, split.y_train.astype(int))
    pred = clf.predict(split.x_test)
    y_true = split.y_test.astype(int).to_numpy()
    accuracy = float(accuracy_score(y_true, pred))
    f1_macro = float(f1_score(y_true, pred, average="macro", zero_division=0))
    test_metrics: dict[str, float] = {"accuracy": accuracy, "f1_macro": f1_macro}
    cm_payload = _confusion_matrix_payload(y_true, pred)

    return clf, test_metrics, cm_payload


def _fit_regression(
    *,
    split: XYSplit,
) -> tuple[RandomForestRegressor, dict[str, float], np.ndarray, np.ndarray]:
    reg = RandomForestRegressor(
        n_estimators=_RF_TREES,
        max_depth=_RF_DEPTH,
        random_state=_RF_SEED,
        n_jobs=-1,
    )
    y_train_f = split.y_train.astype(float)
    y_test_f = split.y_test.astype(float)
    reg.fit(split.x_train, y_train_f)
    pred = reg.predict(split.x_test)
    y_true = y_test_f.to_numpy()
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    mae = float(mean_absolute_error(y_true, pred))
    r2 = float(r2_score(y_true, pred))
    test_metrics = {"rmse": rmse, "mae": mae, "r2": r2}

    return reg, test_metrics, y_true, pred


def run_sklearn_training_pipeline(
    *,
    dataset_path: Path,
    target_column: str,
    model_dir: Path,
) -> dict[str, object]:
    df = _read_csv(dataset_path)
    row_count = int(df.shape[0])

    if target_column not in df.columns:
        msg = f"Колонка таргета «{target_column}» отсутствует в датасете"
        raise ValueError(msg)

    feature_cols = _numeric_feature_columns(df, target_column)

    if not feature_cols:
        msg = "Нет числовых признаков для обучения (кроме таргета)"
        raise ValueError(msg)

    work = df[[target_column, *feature_cols]].copy()
    task, y_series = _infer_task(work[target_column])
    work["_y_"] = y_series
    work = work.dropna(subset=["_y_", *feature_cols])
    cleaned_count = int(work.shape[0])

    if cleaned_count < _MIN_ROWS_AFTER_CLEAN:
        msg = "После очистки осталось слишком мало строк для обучения"
        raise ValueError(msg)

    y = work["_y_"]
    x = work[feature_cols]
    sample_cols = feature_cols[:5]
    sample_summary = {f"{c}__mean": float(x[c].mean()) for c in sample_cols}

    stratify = None

    if task == "classification":
        y_int = y.astype(int)
        class_counts = y_int.value_counts()

        if class_counts.shape[0] < _MIN_CLASSES:
            msg = "Для классификации нужно минимум два класса в данных"
            raise ValueError(msg)

        stratify = y_int

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=_TEST_FRACTION,
        random_state=_RF_SEED,
        stratify=stratify,
    )
    split = XYSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )

    regression_scatter: dict[str, object] | None

    if task == "classification":
        fitted, test_metrics, confusion_payload = _fit_classification(split=split)
        regression_scatter = None
    else:
        fitted, test_metrics, y_true_t, y_pred_t = _fit_regression(split=split)
        confusion_payload = None
        regression_scatter = {
            "actual": [float(x) for x in y_true_t.tolist()],
            "predicted": [float(x) for x in y_pred_t.tolist()],
        }

    viz_frame = work[feature_cols].copy()
    viz_frame[target_column] = work["_y_"].to_numpy(dtype=float)
    viz_frame = viz_frame[[target_column, *feature_cols]]
    correlation_payload = _correlation_matrix_payload(viz_frame)

    importances = np.asarray(fitted.feature_importances_, dtype=float)
    importance_pairs = sorted(
        zip(feature_cols, importances.tolist(), strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )[:_FEATURE_IMPORTANCE_TOP]
    feature_importance = [{"feature": name, "importance": float(imp)} for name, imp in importance_pairs]

    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / "model.joblib"
    joblib.dump(fitted, artifact_path)

    return {
        "task_type": str(task),
        "row_count": row_count,
        "cleaned_row_count": cleaned_count,
        "feature_columns": feature_cols,
        "target_column": target_column,
        "sample_summary": sample_summary,
        "test_metrics": test_metrics,
        "feature_importance": feature_importance,
        "confusion_matrix": confusion_payload,
        "correlation": correlation_payload,
        "regression_scatter": regression_scatter,
        "model_file": str(artifact_path),
    }
