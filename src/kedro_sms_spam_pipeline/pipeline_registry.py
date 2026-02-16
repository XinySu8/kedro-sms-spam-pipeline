"""Project pipelines."""

from kedro_sms_spam_pipeline.pipelines.sms_spam import create_pipeline as create_sms_spam_pipeline
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    sms_spam = create_sms_spam_pipeline()
    return {
        "sms_spam": sms_spam,
        "__default__": sms_spam,
    }
