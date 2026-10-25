from io import BytesIO

import pandas as pd
import pytest

from app.main import infer_task, load_dataframe, run_experiment


def test_load_csv_and_profile_shape():
    frame = load_dataframe("sample.csv", b"age,label\n21,yes\n22,no\n")
    assert frame.shape == (2, 2)
    assert frame.columns.tolist() == ["age", "label"]


def test_infer_numeric_binary_as_classification():
    target = pd.Series([0, 1] * 30)
    assert infer_task(target, "auto") == "classification"


def test_infer_continuous_numeric_as_regression():
    target = pd.Series([float(i) / 7 for i in range(80)])
    assert infer_task(target, "auto") == "regression"


def test_classification_experiment_returns_top_three_and_metrics():
    rows = []
    for i in range(120):
        rows.append({
            "customer_id": f"C{i:03d}",
            "age": 20 + (i % 40),
            "income": 15000 + (i * 137) % 90000,
            "region": ["north", "south", "east"][i % 3],
            "churn": "yes" if (i % 4 == 0 or i % 11 == 0) else "no",
        })
    result = run_experiment(pd.DataFrame(rows), "churn", "classification", 0.2, 7)
    assert result["task"] == "classification"
    assert len(result["top3"]) == 3
    assert "accuracy" in result["top3"][0]
    assert result["file_summary"]["identifier_columns_excluded"] == ["customer_id"]


def test_labeled_tsv_text_parsing():
    frame = load_dataframe("reviews.txt", "positive\tGreat product and fast delivery.\nnegative\tBroken and disappointing item.\n".encode())
    assert frame.columns.tolist() == ["target", "text"]
    assert frame.iloc[0]["target"] == "positive"


def test_regression_experiment_returns_regression_metrics():
    rows = []
    for i in range(100):
        rows.append({"size_sqft": 400 + i * 8, "bedrooms": 1 + i % 4, "area": ["A", "B", "C"][i % 3], "price": 900000 + i * 14000 + (i % 7) * 12000})
    result = run_experiment(pd.DataFrame(rows), "price", "regression", 0.2, 17)
    assert result["task"] == "regression"
    assert len(result["top3"]) == 3
    assert "r2" in result["top3"][0]
    assert "mae" in result["top3"][0]
    assert "accuracy" not in result["top3"][0]


def test_labeled_text_can_train_tfidf_classifier():
    rows = []
    positive = [
        "excellent support and helpful service", "great product and fast delivery", "really good quality and value",
        "happy with the purchase and the result", "works well and arrived quickly", "friendly support solved my issue",
    ]
    negative = [
        "terrible quality and very poor service", "broken product and slow delivery", "bad value and disappointing result",
        "unhappy with the purchase and outcome", "does not work and arrived late", "unhelpful support ignored my issue",
    ]
    for i in range(48):
        rows.append({"review": positive[i % len(positive)] + f" extra comment number {i}", "sentiment": "positive"})
        rows.append({"review": negative[i % len(negative)] + f" additional words {i}", "sentiment": "negative"})
    result = run_experiment(pd.DataFrame(rows), "sentiment", "classification", 0.2, 23)
    assert "review" in result["preprocessing"]["text_columns_tfidf"]
    assert len(result["top3"]) == 3


def test_target_suggestions_prioritize_label_and_exclude_id():
    from app.main import profile_dataframe
    frame = pd.DataFrame({"customer_id": [f"C{i}" for i in range(50)], "age": list(range(20, 70)), "Attrition": ["Yes", "No"] * 25})
    profile = profile_dataframe(frame)
    assert profile["target_suggestions"][0] == "Attrition"
    assert "customer_id" not in profile["target_suggestions"]


def test_unlabeled_raw_text_explains_accuracy_needs_labels():
    text = "\n".join([f"This is a different sentence number {i} with sample words." for i in range(40)])
    frame = load_dataframe("notes.txt", text.encode())
    assert frame.columns.tolist() == ["text"]
    with pytest.raises(ValueError, match="raw text but no target labels"):
        run_experiment(frame, "text", "classification", 0.2, 42)


def test_csv_loader_falls_back_for_legacy_latin1_encoding():
    # The DataCo supply-chain CSV contains a few non-UTF-8 characters.
    payload = "city,note\nPeña,shipping\n".encode("latin-1")
    frame = load_dataframe("legacy.csv", payload)
    assert frame.loc[0, "city"] == "Peña"


def test_large_dataset_limits_cover_dataco_supply_chain():
    from app.main import MAX_BYTES, MAX_ROWS, MAX_TRAINING_ROWS
    assert MAX_BYTES >= 100 * 1024 * 1024
    assert MAX_ROWS >= 180_519
    assert MAX_TRAINING_ROWS <= 50_000


def test_profile_prioritizes_risk_target_and_redacts_identifiers():
    from app.main import profile_dataframe
    frame = pd.DataFrame({
        "Customer Id": [1001, 1002, 1003],
        "Customer Email": ["a@example.test", "b@example.test", "c@example.test"],
        "Late_delivery_risk": [0, 1, 0],
        "Category Name": ["A", "B", "A"],
    })
    profile = profile_dataframe(frame)
    assert profile["target_suggestions"][0] == "Late_delivery_risk"
    assert profile["preview"][0]["Customer Id"] == "[redacted]"
    assert profile["preview"][0]["Customer Email"] == "[redacted]"
    assert "Customer Id" in profile["preview_redacted_columns"]
