#TODO: model.py

from xgboost import XGBClassifier

def create_model(**kwargs):
    # Initialize model
    xgb_clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42,
        **kwargs,
    )

    return xgb_clf
