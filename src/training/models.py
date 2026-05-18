# models.py
# Defines the three ML models we'll train and their hyperparameters.
# Each model is returned as a dictionary with a name and the model object.
# Having all models in one place makes it easy to add or remove models later.

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_models() -> list[dict]:
    """
    Returns a list of model configs. Each config has:
    - name: used for MLflow logging and display
    - model: the actual scikit-learn/xgboost model object
    - params: the hyperparameters we'll log to MLflow

    Why these three models?
    - LogisticRegression: simple, fast, interpretable — good baseline
    - RandomForest: ensemble of decision trees — handles non-linear patterns
    - XGBoost: gradient boosting — usually the best performer on tabular data

    Key parameters explained:
    - class_weight='balanced': tells the model fraud is rare —
      automatically weights fraud examples higher during training
    - n_estimators: number of trees in the forest/ensemble
    - max_depth: how deep each tree can grow — deeper = more complex
    - random_state=42: makes results reproducible
    - eval_metric='logloss': XGBoost's internal scoring during training
    """
    return [
        {
            'name': 'LogisticRegression',
            'model': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            ),
            'params': {
                'class_weight': 'balanced',
                'max_iter': 1000
            }
        },
        {
            'name': 'RandomForest',
            'model': RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                class_weight='balanced',
                random_state=42
            ),
            'params': {
                'n_estimators': 100,
                'max_depth': 6,
                'class_weight': 'balanced'
            }
        },
        {
            'name': 'XGBoost',
            'model': XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                scale_pos_weight=8,  # ratio of legitimate to fraud (~207/24)
                random_state=42,
                eval_metric='logloss',
                verbosity=0  # suppress XGBoost's own output
            ),
            'params': {
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.1,
                'scale_pos_weight': 8
            }
        }
    ]