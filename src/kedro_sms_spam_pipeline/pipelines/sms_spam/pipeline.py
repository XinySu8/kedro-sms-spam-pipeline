from kedro.pipeline import Pipeline, node, pipeline

from .nodes import ingest_sms_spam

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
            func=ingest_sms_spam,
            inputs="sms_spam_raw",
            outputs="sms_spam_clean",
            name="ingest_sms_spam_node",
            )
        ]
    )



from .nodes import split_sms_spam, featurize_tfidf, train_baseline_model, evaluate_classifier


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_sms_spam,
                inputs=dict(sms_spam_clean="sms_spam_clean", params="params:split"),
                outputs=["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"],
                name="split_sms_spam_node",
            ),
            node(
                func=featurize_tfidf,
                inputs=dict(X_train="X_train", X_val="X_val", X_test="X_test", params="params:tfidf"),
                outputs=["X_train_vec", "X_val_vec", "X_test_vec", "tfidf_vectorizer"],
                name="tfidf_featurize_node",
            ),
            node(
                func=train_baseline_model,
                inputs=dict(X_train_vec="X_train_vec", y_train="y_train", params="params:model"),
                outputs="spam_classifier_model",
                name="train_baseline_model_node",
            ),
            node(
                func=evaluate_classifier,
                inputs=dict(model="spam_classifier_model", X_test_vec="X_test_vec", y_test="y_test"),
                outputs="metrics",
                name="evaluate_model_node",
            ),
        ],
        tags="sms_spam",
    )
