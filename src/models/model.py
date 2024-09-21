#TODO: model.py

from xgboost import XGBClassifier

def create_model():
    # Initialize model
    xgb_clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42,
    )

    return xgb_clf
