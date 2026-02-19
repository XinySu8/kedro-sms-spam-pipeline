import pandas as pd

from kedro_sms_spam_pipeline.pipelines.sms_spam.nodes import (
    split_sms_spam,
    featurize_tfidf,
    train_baseline_model,
    evaluate_classifier,
)


def test_end_to_end_nodes_sanity():
    df = pd.DataFrame(
        {
            "text": [
                "Hi, are we still meeting today?",
                "Congratulations! You won a free prize, click now",
                "Call me when you are free",
                "WIN cash now!!! limited offer",
                "Let's have lunch tomorrow",
                "You have been selected for a gift card, claim now",
            ],
            "label": ["ham", "spam", "ham", "spam", "ham", "spam"],
        }
    )

    split_params = {"test_size": 0.33, "val_size": 0.5, "random_state": 42, "stratify": True}
    tfidf_params = {"max_features": 1000, "ngram_range": [1, 2], "min_df": 1}
    model_params = {"C": 1.0, "max_iter": 200, "class_weight": "balanced"}

    X_train, X_val, X_test, y_train, y_val, y_test = split_sms_spam(df, split_params)
    X_train_vec, X_val_vec, X_test_vec, _vectorizer = featurize_tfidf(X_train, X_val, X_test, tfidf_params)
    model = train_baseline_model(X_train_vec, y_train, model_params)
    metrics = evaluate_classifier(model, X_test_vec, y_test)

    assert "f1" in metrics
    assert 0.0 <= metrics["f1"] <= 1.0
