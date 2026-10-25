from __future__ import annotations

import csv
import json
import os
import re
import time
from io import BytesIO, StringIO
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier, SGDRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

APP_NAME = "AutoML API"
MAX_BYTES = 120 * 1024 * 1024
MAX_ROWS = 250_000
MAX_TRAINING_ROWS = 30_000
MAX_COLUMNS = 250
MAX_TEXT_FEATURES = 3_000
MIN_TRAIN_ROWS = 25

app = FastAPI(
    title=APP_NAME,
    description=(
        "Transparent AutoML prototype for profiling, cleaning, preprocessing, "
        "train/test evaluation, and baseline model comparison."
    ),
    version="1.0.0",
)

_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", _default_origins).split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

ID_NAME_PARTS = {"id", "uuid", "key", "recordid", "rowid", "customerid", "patientid", "memberid", "acoid"}


def _name_tokens(name: str) -> list[str]:
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    return [token for token in re.split(r"[^a-zA-Z0-9]+", camel_split.lower()) if token]


@app.get("/")
def root() -> dict[str, str]:
    return {"name": APP_NAME, "status": "ok", "docs": "/docs"}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": APP_NAME}


def _safe_filename(name: str | None) -> str:
    return os.path.basename(name or "dataset").replace("\x00", "")


def load_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    """Read supported data formats without saving uploaded contents to disk."""
    suffix = os.path.splitext(filename.lower())[1]
    try:
        if suffix in {".csv", ".tsv"}:
            if suffix == ".tsv":
                separator = "\t"
            else:
                # Sniff only a small prefix, then use pandas' faster C parser. The
                # Python engine used for sep=None is painfully slow on 100k+ rows.
                sample = content[:64 * 1024].decode("utf-8-sig", errors="replace")
                try:
                    separator = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
                except csv.Error:
                    separator = ","
            csv_args = {
                "sep": separator,
                "engine": "c",
                "low_memory": False,
            }
            try:
                frame = pd.read_csv(BytesIO(content), encoding="utf-8-sig", **csv_args)
            except UnicodeDecodeError:
                # Public data-science datasets are sometimes exported using a
                # Windows-1252/Latin-1 encoding. Retry without losing accented text.
                frame = pd.read_csv(BytesIO(content), encoding="latin-1", **csv_args)
        elif suffix in {".xlsx"}:
            frame = pd.read_excel(BytesIO(content), engine="openpyxl")
        elif suffix == ".json":
            text = content.decode("utf-8-sig")
            try:
                obj = json.loads(text)
                if isinstance(obj, list):
                    frame = pd.DataFrame(obj)
                elif isinstance(obj, dict):
                    if isinstance(obj.get("data"), list):
                        frame = pd.DataFrame(obj["data"])
                    else:
                        frame = pd.json_normalize(obj)
                else:
                    raise ValueError("JSON must contain an array of records or an object.")
            except json.JSONDecodeError:
                frame = pd.read_json(StringIO(text), lines=True)
        elif suffix == ".txt":
            text = content.decode("utf-8-sig", errors="replace")
            lines = [line for line in text.splitlines() if line.strip()]
            if not lines:
                raise ValueError("The text file is empty.")
            # Labeled unstructured text: one example per line, formatted as label<TAB>text.
            if all("\t" in line for line in lines):
                records = [line.split("\t", maxsplit=1) for line in lines]
                frame = pd.DataFrame(records, columns=["target", "text"])
            else:
                frame = pd.DataFrame({"text": lines})
        else:
            raise ValueError("Unsupported format. Upload CSV, TSV, XLSX, JSON, or TXT.")
    except Exception as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Unsupported format"):
            raise
        raise ValueError(f"Could not read this dataset: {exc}") from exc

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("The uploaded file contains no usable rows.")
    if len(frame) > MAX_ROWS:
        raise ValueError(f"This prototype supports at most {MAX_ROWS:,} rows per upload.")
    if len(frame.columns) > MAX_COLUMNS:
        raise ValueError(f"This prototype supports at most {MAX_COLUMNS} columns per upload.")

    frame.columns = [str(col).strip() for col in frame.columns]
    if any(not col for col in frame.columns):
        frame.columns = [col if col else f"column_{i + 1}" for i, col in enumerate(frame.columns)]
    if not frame.columns.is_unique:
        raise ValueError("The dataset has duplicate column names. Rename duplicate headers and upload again.")
    # Normalize infinities to missing values before profiling/preprocessing.
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def _looks_like_text_column(name: str, series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    normalized = re.sub(r"[^a-z0-9]", "", str(name).lower())
    text_name_hints = ("text", "review", "comment", "description", "message", "sentence", "document", "content", "transcript", "feedback", "notes")
    if any(hint in normalized for hint in text_name_hints):
        return True
    values = series.dropna().astype(str)
    if values.empty:
        return False
    lengths = values.str.len()
    return bool(float(lengths.median()) >= 35 or (float(lengths.mean()) >= 24 and float(lengths.max()) >= 80))


def _column_kind(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return "numeric"
    if _looks_like_text_column(str(series.name or ""), series):
        return "text"
    values = series.dropna().astype(str)
    if values.empty:
        return "empty"
    lengths = values.str.len()
    if float(lengths.median()) >= 35 or (float(lengths.mean()) >= 24 and float(lengths.max()) >= 80):
        return "text"
    return "categorical"


def _json_value(value: Any) -> Any:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _should_redact_preview_column(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", str(name).lower())
    tokens = _name_tokens(name)
    sensitive_parts = (
        "email", "password", "phone", "address", "street", "fname", "lname",
        "firstname", "lastname", "fullname", "username", "zipcode", "postalcode",
        "accountnumber", "ssn", "socialsecurity", "token", "secret",
    )
    return (
        any(part in normalized for part in sensitive_parts)
        or normalized in ID_NAME_PARTS
        or (bool(tokens) and tokens[-1] in {"id", "uuid", "key"})
    )


def profile_dataframe(frame: pd.DataFrame) -> dict[str, Any]:
    column_profiles = []
    for name in frame.columns:
        series = frame[name]
        non_null = series.dropna()
        column_profiles.append({
            "name": str(name),
            "dtype": str(series.dtype),
            "kind": _column_kind(series),
            "missing": int(series.isna().sum()),
            "missing_percent": round(float(series.isna().mean() * 100), 2),
            "unique": int(series.nunique(dropna=True)),
        })
    text_cols = [col["name"] for col in column_profiles if col["kind"] == "text"]
    numeric_cols = [col["name"] for col in column_profiles if col["kind"] == "numeric"]
    categorical_cols = [col["name"] for col in column_profiles if col["kind"] == "categorical"]
    rows = frame.head(12).to_dict(orient="records")
    preview = [{
        str(k): ("[redacted]" if _should_redact_preview_column(str(k)) and pd.notna(v) else _json_value(v))
        for k, v in row.items()
    } for row in rows]
    preview_redacted_columns = [str(col) for col in frame.columns if _should_redact_preview_column(str(col))]

    priority_targets = {"target", "label", "class", "outcome", "y", "churn", "attrition", "diagnosis", "species", "category", "default", "survived", "loanstatus", "isfraud", "fraud", "purchased", "clicked", "latedeliveryrisk", "deliveryrisk"}
    preferred_targets = []
    plausible_targets = []
    for col in column_profiles:
        name = re.sub(r"[^a-z0-9]", "", col["name"].lower())
        tokens = _name_tokens(col["name"])
        identifier_like = name in ID_NAME_PARTS or (bool(tokens) and tokens[-1] in {"id", "uuid"}) or name in {"email", "phone", "phonenumber", "username", "fullname", "address", "accountnumber"}
        if name in priority_targets and col["unique"] > 1 and col["missing_percent"] < 40:
            preferred_targets.append(col["name"])
        elif not identifier_like and 1 < col["unique"] <= max(50, int(len(frame) * 0.25)) and col["missing_percent"] < 40:
            plausible_targets.append(col["name"])
    target_suggestions = preferred_targets + [name for name in plausible_targets if name not in preferred_targets]
    data_kind = "Text-heavy / mixed" if text_cols else "Structured tabular"
    if len(frame.columns) == 1 and text_cols:
        data_kind = "Unstructured text"
    return {
        "rows": int(len(frame)),
        "columns_count": int(len(frame.columns)),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "text_columns": text_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "data_kind": data_kind,
        "columns": column_profiles,
        "target_suggestions": target_suggestions,
        "preview": preview,
        "preview_redacted_columns": preview_redacted_columns,
    }


async def read_upload(file: UploadFile) -> tuple[str, bytes, pd.DataFrame]:
    filename = _safe_filename(file.filename)
    content = await file.read(MAX_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 120 MB prototype limit.")
    try:
        frame = load_dataframe(filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return filename, content, frame


@app.post("/api/inspect")
async def inspect_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    filename, _content, frame = await read_upload(file)
    profile = profile_dataframe(frame)
    if filename.lower().endswith(".txt") and len(frame.columns) == 1:
        profile["data_kind"] = "Unstructured text"
        profile["text_columns"] = [str(frame.columns[0])]
        for column in profile["columns"]:
            column["kind"] = "text"
    return {"file_name": filename, **profile}


def infer_task(target: pd.Series, task: str) -> str:
    if task in {"classification", "regression"}:
        return task
    if task != "auto":
        raise ValueError("Task must be auto, classification, or regression.")
    as_numeric = pd.to_numeric(target, errors="coerce")
    numeric_ratio = float(as_numeric.notna().mean()) if len(target) else 0.0
    unique_count = int(target.nunique(dropna=True))
    small_cardinality = unique_count <= max(10, int(len(target) * 0.02))
    if numeric_ratio >= 0.95 and not small_cardinality:
        return "regression"
    return "classification"


def is_identifier_column(name: str, series: pd.Series) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    tokens = _name_tokens(name)
    # Column names that are identifiers remain identifiers even when customer/order
    # records repeat. Requiring high cardinality missed common columns in supply-chain data.
    exact_name = normalized in ID_NAME_PARTS or (bool(tokens) and tokens[-1] in {"id", "uuid"})
    key_name = bool(tokens) and tokens[-1] in {"key", "number"}
    private_or_personal_key = normalized in {
        "email", "customeremail", "phone", "phonenumber", "customerphone",
        "username", "fullname", "address", "accountnumber", "customerfname",
        "customerlname", "customerstreet", "customerpassword", "customerzipcode",
    }
    return bool(exact_name or key_name or private_or_personal_key)


def detect_feature_groups(X: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    numeric = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    text: list[str] = []
    for col in X.columns:
        if col in numeric:
            continue
        if _looks_like_text_column(str(col), X[col]):
            text.append(col)
    categorical = [col for col in X.columns if col not in numeric and col not in text]
    return numeric, categorical, text


def build_pipeline(X: pd.DataFrame, estimator) -> tuple[Pipeline, dict[str, list[str]]]:
    numeric_cols, categorical_cols, text_cols = detect_feature_groups(X)
    transformers: list[tuple[str, Any, Any]] = []
    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler(with_mean=False)),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_cols))
    for index, col in enumerate(text_cols):
        # String coercion plus blank replacement is deterministic and doesn't learn from the test data.
        X[col] = X[col].fillna("").astype(str)
        transformers.append((
            f"text_{index}",
            TfidfVectorizer(max_features=MAX_TEXT_FEATURES, ngram_range=(1, 2), stop_words="english", sublinear_tf=True),
            col,
        ))
    if not transformers:
        raise ValueError("No usable feature columns remain after cleaning.")
    preprocess = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
    return Pipeline([("preprocess", preprocess), ("model", estimator)]), {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "text": text_cols,
    }


def _model_candidates(task: str, seed: int) -> dict[str, Any]:
    if task == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1800, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1),
            "Extra Trees": ExtraTreesClassifier(n_estimators=100, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1),
            "SGD Classifier": SGDClassifier(loss="log_loss", max_iter=1800, class_weight="balanced", random_state=seed, tol=1e-3),
        }
    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=seed, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, min_samples_leaf=2, random_state=seed, n_jobs=-1),
        "SGD Regressor": SGDRegressor(max_iter=1800, random_state=seed, tol=1e-3),
    }


def _reason_for_model(name: str, task: str) -> str:
    reasons = {
        "Logistic Regression": "A useful, relatively interpretable linear baseline. It is fast and often strong when features separate classes cleanly.",
        "Random Forest": "Captures nonlinear relationships and feature interactions, while averaging many trees to reduce the variance of a single tree.",
        "Extra Trees": "Uses randomized tree splits to capture nonlinear patterns; its extra randomness can generalize well on mixed tabular features.",
        "SGD Classifier": "Efficient for sparse, high-dimensional feature spaces such as TF-IDF text features; its result can vary with data quality and tuning.",
        "Ridge Regression": "A regularized linear model that is quick to train and useful when many features contribute additively.",
        "SGD Regressor": "A scalable linear candidate that can work efficiently on sparse or high-dimensional features.",
    }
    if name in reasons:
        return reasons[name]
    if task == "regression":
        return "A tree-ensemble candidate that can learn nonlinear relationships; inspect MAE/RMSE as well as R² before choosing it."
    return "A tree-ensemble candidate that can learn nonlinear feature interactions; review class balance and weighted F1 alongside accuracy."


def _feature_importance(pipeline: Pipeline, limit: int = 12) -> list[dict[str, Any]]:
    model = pipeline.named_steps["model"]
    try:
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
    except Exception:
        return []
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_).ravel()
    elif hasattr(model, "coef_"):
        values = np.asarray(model.coef_)
        values = np.mean(np.abs(values), axis=0) if values.ndim > 1 else np.abs(values)
    else:
        return []
    if len(values) != len(names):
        return []
    indices = np.argsort(values)[::-1][:limit]
    result = []
    for index in indices:
        raw_name = str(names[index])
        pretty = re.sub(r"^(numeric|categorical|text_\d+)__", "", raw_name)
        result.append({"feature": pretty[:60], "importance": round(float(values[index]), 6)})
    return result


def run_experiment(
    original: pd.DataFrame,
    target: str,
    requested_task: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    if target not in original.columns:
        raise ValueError("The selected target column is not in the uploaded dataset.")
    if not 0.15 <= test_size <= 0.35:
        raise ValueError("Test-set fraction must be between 0.15 and 0.35.")

    initial_rows = len(original)
    initial_missing = int(original.isna().sum().sum())
    duplicate_rows_removed = int(original.duplicated().sum())
    data = original.drop_duplicates().copy()
    all_empty_columns = [str(col) for col in data.columns if data[col].isna().all()]
    if all_empty_columns:
        data = data.drop(columns=all_empty_columns)
    if target not in data.columns:
        raise ValueError("The target column is entirely empty and cannot be used.")

    target_missing = data[target].isna()
    if data[target].dtype == object or pd.api.types.is_string_dtype(data[target]):
        target_missing = target_missing | data[target].astype("string").str.strip().eq("").fillna(True)
    dropped_target_rows = int(target_missing.sum())
    data = data.loc[~target_missing].copy()
    if len(data) < MIN_TRAIN_ROWS:
        raise ValueError(f"Only {len(data)} labeled rows remain. Use at least {MIN_TRAIN_ROWS} usable rows for a meaningful comparison.")

    actual_task = infer_task(data[target], requested_task)
    y = data.pop(target)
    if actual_task == "regression":
        y = pd.to_numeric(y, errors="coerce")
        valid = y.notna()
        dropped_bad_target = int((~valid).sum())
        data, y = data.loc[valid].copy(), y.loc[valid].astype(float)
        if len(y) < MIN_TRAIN_ROWS or y.nunique() < 2:
            raise ValueError("Regression needs at least 25 valid rows and two or more numeric target values. Choose classification if the target is a category.")
    else:
        y = y.astype(str).str.strip()
        valid = y.ne("")
        dropped_bad_target = int((~valid).sum())
        data, y = data.loc[valid].copy(), y.loc[valid]
        if y.nunique() < 2:
            raise ValueError("Classification needs at least two distinct target classes.")

    identifier_columns = [col for col in data.columns if is_identifier_column(str(col), data[col])]
    if identifier_columns:
        data = data.drop(columns=identifier_columns)
    all_empty_after_target = [str(col) for col in data.columns if data[col].isna().all()]
    if all_empty_after_target:
        data = data.drop(columns=all_empty_after_target)
    if len(data.columns) == 0:
        if len(original.columns) == 1 and _column_kind(original.iloc[:, 0]) == "text":
            raise ValueError("This upload contains raw text but no target labels/features. Supervised model accuracy needs labeled examples. For text classification, upload CSV with a text column plus a label/target column, or TXT with one label<TAB>text example per line.")
        raise ValueError("No usable feature columns remain after the target and identifier columns were removed.")

    cleaned_rows_before_sampling = len(data)
    rows_sampled_out = 0
    known_leakage_columns: list[str] = []
    target_normalized = re.sub(r"[^a-z0-9]", "", str(target).lower())
    # In the widely used DataCo supply-chain dataset, these fields are available only
    # after a shipment/delivery outcome and can reveal Late_delivery_risk. Excluding
    # them avoids an impressively high but misleading score for a prediction-at-order-time task.
    if target_normalized == "latedeliveryrisk":
        leaking_names = {"deliverystatus", "daysforshippingreal", "orderstatus", "shippingdatedateorders"}
        known_leakage_columns = [
            col for col in data.columns
            if re.sub(r"[^a-z0-9]", "", str(col).lower()) in leaking_names
        ]
        if known_leakage_columns:
            data = data.drop(columns=known_leakage_columns)

    # Keep profiling on the full uploaded table, but cap model comparison to a
    # deterministic, stratified sample so a 180k-row CSV remains practical on a laptop.
    if len(data) > MAX_TRAINING_ROWS:
        all_positions = np.arange(len(data))
        stratify_for_sample = None
        if actual_task == "classification" and y.nunique() > 1 and y.value_counts().min() >= 2:
            stratify_for_sample = y
        try:
            selected_positions, _unused_positions = train_test_split(
                all_positions,
                train_size=MAX_TRAINING_ROWS,
                random_state=int(random_state),
                stratify=stratify_for_sample,
            )
        except ValueError:
            selected_positions, _unused_positions = train_test_split(
                all_positions,
                train_size=MAX_TRAINING_ROWS,
                random_state=int(random_state),
            )
        data = data.iloc[selected_positions].copy()
        y = y.iloc[selected_positions].copy()
        rows_sampled_out = cleaned_rows_before_sampling - len(data)

    # Column-specific preprocessing pipelines handle the remaining missing values from the training split.
    text_cols = detect_feature_groups(data)[2]
    for col in text_cols:
        data[col] = data[col].fillna("").astype(str)

    class_counts = y.value_counts() if actual_task == "classification" else None
    stratify = None
    if actual_task == "classification" and class_counts is not None:
        n_test = int(np.ceil(len(y) * test_size))
        n_train = len(y) - n_test
        if int(class_counts.min()) >= 2 and n_test >= len(class_counts) and n_train >= len(class_counts):
            stratify = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            data,
            y,
            test_size=test_size,
            random_state=int(random_state),
            stratify=stratify,
        )
    except ValueError as exc:
        raise ValueError(f"Could not make the train/test split: {exc}") from exc

    if actual_task == "classification":
        candidates: dict[str, Any] = {"Dummy Baseline": DummyClassifier(strategy="most_frequent")}
    else:
        candidates = {"Dummy Baseline": DummyRegressor(strategy="mean")}
    candidates.update(_model_candidates(actual_task, int(random_state)))

    leaderboard: list[dict[str, Any]] = []
    fitted_models: dict[str, tuple[Pipeline, dict[str, Any]]] = {}
    skipped: list[dict[str, str]] = []
    pipeline_feature_groups: dict[str, list[str]] = {"numeric": [], "categorical": [], "text": []}

    for name, estimator in candidates.items():
        start = time.perf_counter()
        try:
            # build_pipeline mutates text columns to strings; pass copies to avoid cross-candidate state.
            pipeline, groups = build_pipeline(X_train.copy(), estimator)
            pipeline.fit(X_train.copy(), y_train)
            predictions = pipeline.predict(X_test.copy())
            elapsed = round(time.perf_counter() - start, 3)
            if actual_task == "classification":
                row = {
                    "model": name,
                    "is_baseline": name == "Dummy Baseline",
                    "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
                    "weighted_f1": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 6),
                    "balanced_accuracy": round(float(balanced_accuracy_score(y_test, predictions)), 6),
                    "precision_weighted": round(float(precision_score(y_test, predictions, average="weighted", zero_division=0)), 6),
                    "recall_weighted": round(float(recall_score(y_test, predictions, average="weighted", zero_division=0)), 6),
                    "fit_seconds": elapsed,
                }
            else:
                rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
                row = {
                    "model": name,
                    "is_baseline": name == "Dummy Baseline",
                    "r2": round(float(r2_score(y_test, predictions)), 6),
                    "mae": round(float(mean_absolute_error(y_test, predictions)), 6),
                    "rmse": round(rmse, 6),
                    "fit_seconds": elapsed,
                }
            leaderboard.append(row)
            fitted_models[name] = (pipeline, row)
            if name != "Dummy Baseline":
                pipeline_feature_groups = groups
        except Exception as exc:
            skipped.append({"model": name, "reason": str(exc)[:220]})

    if not fitted_models:
        raise ValueError("No model trained successfully. Check your target column and data types, then try again.")
    if len(fitted_models) == 1:
        raise ValueError("Only the dummy baseline trained. The feature preprocessing or selected target needs review.")

    rank_key = "accuracy" if actual_task == "classification" else "r2"
    leaderboard.sort(key=lambda item: item.get(rank_key, float("-inf")), reverse=True)
    non_baseline = [item for item in leaderboard if not item["is_baseline"]]
    top3 = non_baseline[:3]
    best = top3[0] if top3 else leaderboard[0]
    best_pipeline = fitted_models[best["model"]][0]

    baseline_row = next((row for row in leaderboard if row["is_baseline"]), None)
    for rank, row in enumerate(top3, 1):
        row["rank"] = rank
        family_reason = _reason_for_model(row["model"], actual_task)
        if actual_task == "classification":
            metric_note = f"Ranked #{rank} by held-out accuracy ({row['accuracy']:.1%}); weighted F1 is {row['weighted_f1']:.3f}."
            if rank == 1 and baseline_row is not None:
                delta = (row["accuracy"] - baseline_row.get("accuracy", 0.0)) * 100
                metric_note += f" This is {delta:+.1f} percentage points versus the dummy baseline."
        else:
            metric_note = f"Ranked #{rank} by held-out R² ({row['r2']:.3f}); MAE is {row['mae']:.3f} and RMSE is {row['rmse']:.3f}."
            if rank == 1 and baseline_row is not None:
                delta = row["r2"] - baseline_row.get("r2", 0.0)
                metric_note += f" R² difference versus dummy baseline: {delta:+.3f}."
        row["reason"] = f"{metric_note} {family_reason}"
    if actual_task == "classification":
        best_predictions = best_pipeline.predict(X_test.copy())
        labels = sorted(pd.Series(y).astype(str).unique().tolist())
        # Keep the matrix readable when a dataset has many classes.
        if len(labels) <= 14:
            matrix = confusion_matrix(y_test, best_predictions, labels=labels).tolist()
            cm = {"labels": labels, "matrix": matrix}
        else:
            cm = None
        counts = pd.Series(y).value_counts().head(12)
        target_distribution = [{"label": str(label), "count": int(count)} for label, count in counts.items()]
    else:
        cm = None
        try:
            target_buckets = pd.cut(y, bins=min(10, int(y.nunique())), duplicates="drop")
            counts = target_buckets.value_counts(sort=False).head(12)
            target_distribution = [{"label": str(label), "count": int(count)} for label, count in counts.items()]
        except Exception:
            target_distribution = []

    missing_by_column = original.isna().mean().mul(100).sort_values(ascending=False).head(10)
    missing_chart = [{"column": str(col), "missing_percent": round(float(pct), 2)} for col, pct in missing_by_column.items()]
    row_count_after_cleaning = cleaned_rows_before_sampling
    class_imbalance_ratio = None
    warnings: list[str] = []
    if actual_task == "classification" and class_counts is not None and len(class_counts) > 1:
        class_imbalance_ratio = round(float(class_counts.max() / max(1, class_counts.min())), 2)
        if class_imbalance_ratio >= 2:
            warnings.append("The target is imbalanced. Accuracy alone may overstate performance; compare weighted F1, balanced accuracy, precision and recall.")
    if dropped_target_rows or dropped_bad_target:
        warnings.append("Rows with missing or invalid targets were removed because a supervised model cannot learn their correct output.")
    if identifier_columns:
        warnings.append("ID-like and personal-identifying columns were excluded to reduce memorization and avoid using direct identifiers as model features.")
    if known_leakage_columns:
        warnings.append("For Late_delivery_risk, post-outcome fields were excluded to reduce label leakage: " + ", ".join(known_leakage_columns) + ".")
    if rows_sampled_out:
        warnings.append(f"The upload contained {cleaned_rows_before_sampling:,} usable rows. To keep training practical, metrics were evaluated on a reproducible sample of {len(data):,} rows; dataset profiling still covered the full upload.")
    warnings.append("Holdout results are estimates, not a guarantee of future performance. Check leakage and validate the selected model on new data before real-world use.")

    return {
        "file_summary": {
            "original_rows": int(initial_rows),
            "rows_after_cleaning": int(row_count_after_cleaning),
            "rows_used_for_model_comparison": int(len(data)),
            "rows_sampled_out_for_training": int(rows_sampled_out),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "original_columns": int(len(original.columns)),
            "feature_columns_used": int(data.shape[1]),
            "missing_cells_before": initial_missing,
            "duplicate_rows_removed": duplicate_rows_removed,
            "target_rows_removed": int(dropped_target_rows + dropped_bad_target),
            "empty_columns_removed": all_empty_columns,
            "identifier_columns_excluded": identifier_columns,
        },
        "task": actual_task,
        "target": target,
        "test_size": test_size,
        "random_state": int(random_state),
        "primary_metric": "Accuracy" if actual_task == "classification" else "R²",
        "leaderboard": leaderboard,
        "top3": top3,
        "best_model": best["model"],
        "best_score": best.get(rank_key),
        "class_imbalance_ratio": class_imbalance_ratio,
        "preprocessing": {
            "numeric_columns": pipeline_feature_groups["numeric"],
            "categorical_columns": pipeline_feature_groups["categorical"],
            "text_columns_tfidf": pipeline_feature_groups["text"],
            "steps": [
                f"Removed {duplicate_rows_removed:,} exact duplicate rows.",
                f"Removed {len(all_empty_columns)} completely empty column(s).",
                f"Removed {dropped_target_rows + dropped_bad_target:,} row(s) with missing/invalid target values.",
                f"Excluded {len(known_leakage_columns)} known post-outcome leakage column(s) for this target.",
                f"Used a deterministic sample of {len(data):,} rows for model comparison out of {cleaned_rows_before_sampling:,} cleaned rows.",
                "Numeric columns: median imputation and scaling.",
                "Categorical columns: most-frequent imputation and one-hot encoding.",
                "Long text columns: TF-IDF vectorization; text blanks become empty strings.",
                "Split into train/test before fitting preprocessing to reduce data leakage.",
            ],
        },
        "charts": {
            "model_comparison": [
                {"model": row["model"], "score": row.get(rank_key, 0), "is_baseline": row["is_baseline"]}
                for row in leaderboard
            ],
            "missing_by_column": missing_chart,
            "target_distribution": target_distribution,
            "confusion_matrix": cm,
            "feature_importance": _feature_importance(best_pipeline),
        },
        "warnings": warnings,
        "skipped_models": skipped,
    }


@app.post("/api/train")
async def train_dataset(
    file: UploadFile = File(...),
    target: str = Form(...),
    task: str = Form("auto"),
    test_size: float = Form(0.2),
    random_state: int = Form(42),
) -> dict[str, Any]:
    _filename, _content, frame = await read_upload(file)
    try:
        return run_experiment(frame, target, task.lower(), float(test_size), int(random_state))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
