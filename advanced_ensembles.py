from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from xgboost import XGBClassifier

RANDOM_STATE = 42

xgb_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        reg_lambda=10,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )),
])

voting_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("vote", VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ("svc", SVC(
                kernel="rbf", probability=True, random_state=RANDOM_STATE,
            )),
            ("rf", RandomForestClassifier(
                n_estimators=100, max_depth=3, random_state=RANDOM_STATE,
            )),
            ("gb", GradientBoostingClassifier(
                n_estimators=100, max_depth=3, random_state=RANDOM_STATE,
            )),
        ],
        voting="soft",
    )),
])
