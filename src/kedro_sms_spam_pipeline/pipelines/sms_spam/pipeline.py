from kedro.pipeline import Pipeline, node

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