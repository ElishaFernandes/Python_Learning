import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing     import StandardScaler, OneHotEncoder
from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer
from sklearn.impute            import SimpleImputer
from sklearn.linear_model      import LogisticRegression
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.metrics           import (
    roc_auc_score, classification_report,
    RocCurveDisplay, ConfusionMatrixDisplay,
    brier_score_loss,
)
from sklearn.calibration       import CalibratedClassifierCV, calibration_curve

# XGBoost is optional — falls back gracefully if not installed
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    print("xgboost not installed — skipping XGBClassifier.")

import joblib, os

# ---------------------------------------------------------------------------
# 0.  CONFIGURATION
# ---------------------------------------------------------------------------

RANDOM_STATE    = 42
RECOVERY_RATE   = 0.10          # 10 % recovery → LGD = 90 %
LGD             = 1 - RECOVERY_RATE
MODEL_SAVE_PATH = "best_pd_model.pkl"

# Columns from Task 3 and 4_Loan_Data.csv
# customer_id is dropped — not predictive
NUMERIC_FEATURES = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score",             # credit score — strong default predictor
]
CATEGORICAL_FEATURES = []     # no categorical columns in this dataset
TARGET = "default"


# ---------------------------------------------------------------------------
# 1.  DATA LOADING & FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def load_and_engineer(filepath: str) -> pd.DataFrame:
    """
    Load raw loan data and create derived features that signal default risk.
    """
    df = pd.read_csv(filepath)

    # Strip whitespace from column names (common in Excel exports)
    df.columns = df.columns.str.strip()
    print(f"\n  Detected columns: {list(df.columns)}")
    print(f"  Shape: {df.shape}  | Default rate: {df[TARGET].mean():.2%}\n")

    # ── Derived ratios (strong predictors in credit risk literature) ───────

    # Debt-to-income: what fraction of income is consumed by total debt
    df["debt_to_income"] = df["total_debt_outstanding"] / (df["income"] + 1e-9)

    # Loan-to-income: size of this specific loan relative to annual income
    df["loan_to_income"] = df["loan_amt_outstanding"] / (df["income"] + 1e-9)

    # Credit utilisation: loan outstanding vs number of credit lines
    df["credit_utilisation"] = df["loan_amt_outstanding"] / (
        df["credit_lines_outstanding"] + 1e-9
    )

    # FICO risk band: below 620 is subprime, above 740 is prime
    # Encoding as a distance-from-prime makes the relationship more linear
    df["fico_below_620"] = (df["fico_score"] < 620).astype(int)
    df["fico_prime"]     = (df["fico_score"] >= 740).astype(int)
    df["fico_distance"]  = df["fico_score"] - 660   # signed distance from median

    # Log-income (right-skewed distribution → compress with log)
    df["log_income"] = np.log1p(df["income"])

    # Short tenure flag: <2 years employed is a common risk signal
    df["short_tenure"] = (df["years_employed"] < 2).astype(int)

    return df


def get_feature_lists(df: pd.DataFrame):
    """Return updated feature lists after engineering."""
    extra_numeric = [
        c for c in [
            "debt_to_income", "loan_to_income", "credit_utilisation",
            "fico_below_620", "fico_prime", "fico_distance",
            "log_income", "short_tenure",
        ]
        if c in df.columns
    ]
    num_feats = [c for c in NUMERIC_FEATURES if c in df.columns] + extra_numeric
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return num_feats, cat_feats


# ---------------------------------------------------------------------------
# 2.  PREPROCESSING PIPELINE
# ---------------------------------------------------------------------------

def build_preprocessor(num_feats, cat_feats):
    """
    Scikit-learn ColumnTransformer:
      - Numeric  : median imputation → standard scaling
      - Categorical: most-frequent imputation → one-hot encoding
    """
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers = []
    if num_feats:
        transformers.append(("num", numeric_pipe, num_feats))
    if cat_feats:
        transformers.append(("cat", categorical_pipe, cat_feats))

    return ColumnTransformer(transformers, remainder="drop")


# ---------------------------------------------------------------------------
# 3.  MODEL ZOO
# ---------------------------------------------------------------------------

def build_models(preprocessor):
    """
    Returns a dict of named sklearn Pipelines, each ending in a classifier.
    All classifiers output calibrated probabilities (predict_proba).
    """
    models = {}

    # ── Logistic Regression (interpretable baseline) ───────────────────────
    models["LogisticRegression"] = Pipeline([
        ("prep",  preprocessor),
        ("clf",   LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    # ── Decision Tree (interpretable, non-linear) ─────────────────────────
    models["DecisionTree"] = Pipeline([
        ("prep", preprocessor),
        ("clf",  DecisionTreeClassifier(
            max_depth=6, class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    # ── Random Forest (ensemble, handles non-linearity + interactions) ─────
    models["RandomForest"] = Pipeline([
        ("prep", preprocessor),
        ("clf",  RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE,
        )),
    ])

    # ── Gradient Boosting (sequential ensemble, often best on tabular data)─
    models["GradientBoosting"] = Pipeline([
        ("prep", preprocessor),
        ("clf",  GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=RANDOM_STATE,
        )),
    ])

    # ── XGBoost (if available) ─────────────────────────────────────────────
    if _HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("prep", preprocessor),
            ("clf",  XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="logloss",
                scale_pos_weight=10,   # handles class imbalance
                random_state=RANDOM_STATE,
            )),
        ])

    return models


# ---------------------------------------------------------------------------
# 4.  TRAINING, CROSS-VALIDATION & COMPARISON
# ---------------------------------------------------------------------------

def evaluate_models(models, X_train, y_train, cv=5):
    """
    Stratified k-fold CV; returns a DataFrame sorted by mean ROC-AUC.
    ROC-AUC is preferred over accuracy for imbalanced default datasets.
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, model in models.items():
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_splitter, scoring="roc_auc", n_jobs=-1,
        )
        results[name] = {"mean_auc": scores.mean(), "std_auc": scores.std()}
        print(f"  {name:<22}  AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    return pd.DataFrame(results).T.sort_values("mean_auc", ascending=False)


# ---------------------------------------------------------------------------
# 5.  CALIBRATION  (critical for probability outputs used in EL calculations)
# ---------------------------------------------------------------------------

def calibrate_model(model, X_cal, y_cal):
    """
    Wrap the fitted model in Platt scaling so probabilities are well-calibrated.
    Without this, tree ensembles often return overconfident/underconfident PDs.
    """
    return CalibratedClassifierCV(model, cv="prefit", method="sigmoid").fit(X_cal, y_cal)


# ---------------------------------------------------------------------------
# 6.  FEATURE IMPORTANCE
# ---------------------------------------------------------------------------

def plot_feature_importance(model, feature_names: list, top_n: int = 15):
    """Extract and plot feature importances from tree-based models."""
    clf = model.named_steps["clf"]

    # Handle calibrated wrapper
    if hasattr(clf, "estimator"):
        clf = clf.estimator

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        print("  Model does not expose feature importances — skipping.")
        return

    # The preprocessor transforms features; get transformed names if possible
    try:
        prep = model.named_steps["prep"]
        ohe_names = []
        if "cat" in prep.named_transformers_:
            ohe_names = list(
                prep.named_transformers_["cat"]
                    .named_steps["ohe"]
                    .get_feature_names_out(
                        [c for c in CATEGORICAL_FEATURES if c in feature_names]
                    )
            )
        num_names = [c for c in feature_names if c not in CATEGORICAL_FEATURES]
        all_names = num_names + ohe_names
    except Exception:
        all_names = [f"f{i}" for i in range(len(importances))]

    n = min(top_n, len(importances))
    idx = np.argsort(importances)[-n:]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [all_names[i] if i < len(all_names) else f"f{i}" for i in idx],
        importances[idx],
        color="steelblue",
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {n} Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()
    print("  Saved feature_importance.png")


# ---------------------------------------------------------------------------
# 7.  FULL TRAINING PIPELINE
# ---------------------------------------------------------------------------

def train(filepath: str):
    """
    End-to-end pipeline: load → engineer → split → compare → fit → calibrate.
    Returns the best calibrated model and the feature lists used.
    """
    print("\n" + "="*60)
    print("  LOADING & ENGINEERING DATA")
    print("="*60)
    df = load_and_engineer(filepath)
    num_feats, cat_feats = get_feature_lists(df)
    all_feats = num_feats + cat_feats

    print(f"  Rows: {len(df):,}  |  Default rate: {df[TARGET].mean():.2%}")
    print(f"  Numeric features : {num_feats}")
    print(f"  Categorical features: {cat_feats}")

    X = df[all_feats]
    y = df[TARGET]

    # Stratified split — keep class balance in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    # Hold out a slice for post-fit calibration
    X_train_fit, X_cal, y_train_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )

    print("\n" + "="*60)
    print("  MODEL COMPARISON  (5-fold stratified CV on training set)")
    print("="*60)
    preprocessor = build_preprocessor(num_feats, cat_feats)
    models       = build_models(preprocessor)
    results_df   = evaluate_models(models, X_train_fit, y_train_fit)

    best_name    = results_df.index[0]
    print(f"\n  ► Best model: {best_name}  (AUC = {results_df.iloc[0]['mean_auc']:.4f})\n")

    # Fit best model on full training set (train_fit + cal, not test)
    best_model   = models[best_name]
    best_model.fit(X_train, y_train)

    # Calibrate on the held-out calibration slice
    print("  Calibrating probabilities...")
    calibrated   = CalibratedClassifierCV(best_model, cv="prefit", method="sigmoid")
    calibrated.fit(X_cal, y_cal)

    # ── Evaluation on test set ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("  HOLD-OUT TEST SET EVALUATION")
    print("="*60)
    y_pred_proba = calibrated.predict_proba(X_test)[:, 1]
    y_pred       = (y_pred_proba >= 0.5).astype(int)

    print(f"  ROC-AUC          : {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"  Brier Score      : {brier_score_loss(y_test, y_pred_proba):.4f}  (lower = better calibration)")
    print("\n  Classification Report (threshold = 0.50):")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

    # ── Calibration curve (reliability diagram) ───────────────────────────
    frac_pos, mean_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, frac_pos, "s-", label=best_name)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("calibration_curve.png", dpi=150)
    plt.show()

    # ── Feature importance ─────────────────────────────────────────────────
    plot_feature_importance(best_model, all_feats)

    # Persist the calibrated model
    joblib.dump({"model": calibrated, "features": all_feats}, MODEL_SAVE_PATH)
    print(f"\n  Model saved to '{MODEL_SAVE_PATH}'")

    return calibrated, all_feats, X_test, y_test


# ---------------------------------------------------------------------------
# 8.  INFERENCE — EXPECTED LOSS
# ---------------------------------------------------------------------------

def load_model(path: str = MODEL_SAVE_PATH):
    """Load a persisted model bundle."""
    bundle = joblib.load(path)
    return bundle["model"], bundle["features"]


def predict_expected_loss(
    loan: dict,
    model=None,
    features: list = None,
    recovery_rate: float = RECOVERY_RATE,
) -> dict:
    """
    Given a dictionary of loan/borrower characteristics, return:
        PD  — probability of default
        LGD — loss given default (= 1 - recovery_rate)
        EAD — exposure at default (= loan_amt_outstanding)
        EL  — expected loss = PD × LGD × EAD

    Parameters
    ----------
    loan          : dict with keys matching the feature columns
    model         : trained (calibrated) sklearn pipeline; if None, loads from disk
    features      : list of features the model was trained on
    recovery_rate : fraction recovered in default (0.10 = 10 %)

    Returns
    -------
    dict with keys: PD, LGD, EAD, EL
    """
    if model is None:
        model, features = load_model()

    # Build a single-row DataFrame and apply the same feature engineering
    row = pd.DataFrame([loan])

    # Re-apply the same feature engineering as training
    row["debt_to_income"]    = row["total_debt_outstanding"] / (row["income"] + 1e-9)
    row["loan_to_income"]    = row["loan_amt_outstanding"]   / (row["income"] + 1e-9)
    row["credit_utilisation"]= row["loan_amt_outstanding"]   / (row["credit_lines_outstanding"] + 1e-9)
    row["log_income"]        = np.log1p(row["income"])
    row["short_tenure"]      = (row["years_employed"] < 2).astype(int)
    if "fico_score" in row.columns:
        row["fico_below_620"] = (row["fico_score"] < 620).astype(int)
        row["fico_prime"]     = (row["fico_score"] >= 740).astype(int)
        row["fico_distance"]  = row["fico_score"] - 660

    # Keep only the features the model was trained on
    row = row.reindex(columns=features, fill_value=np.nan)

    pd_val = float(model.predict_proba(row)[0, 1])
    lgd    = 1 - recovery_rate
    ead    = float(loan.get("loan_amt_outstanding", 0))
    el     = pd_val * lgd * ead

    result = {"PD": pd_val, "LGD": lgd, "EAD": ead, "EL": el}
    return result


# ---------------------------------------------------------------------------
# 9.  DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    # ── Generate synthetic data if no CSV is available ────────────────────
    DATA_PATH = "/Users/Elisha/Desktop/Python learning/JPMorgan/Task 3 and 4_Loan_Data.csv"
    if not os.path.exists(DATA_PATH):
        print("  WARNING: Real CSV not found — generating synthetic dataset for demo...")
        np.random.seed(RANDOM_STATE)
        n = 5_000

        income       = np.random.lognormal(mean=10.5, sigma=0.6, size=n)
        debt         = income * np.random.uniform(0.1, 3.0, size=n)
        loan_amt     = income * np.random.uniform(0.05, 1.5, size=n)
        years_emp    = np.random.exponential(scale=5, size=n).clip(0, 30)
        credit_lines = np.random.randint(1, 20, size=n)
        # FICO scores ~ Normal(680, 80), clipped to realistic range [300, 850]
        fico         = np.random.normal(loc=680, scale=80, size=n).clip(300, 850).astype(int)

        # True PD logistic function — higher debt, lower FICO → higher default risk
        log_odds = (
            -4.0
            + 1.5  * (debt / (income + 1))
            - 0.8  * np.log1p(income)
            - 0.5  * years_emp
            - 0.01 * (fico - 660)          # each point above 660 lowers risk
            + 0.3  * (loan_amt / (income + 1))
        )
        true_pd      = 1 / (1 + np.exp(-log_odds))
        default_flag = (np.random.rand(n) < true_pd).astype(int)

        synthetic_df = pd.DataFrame({
            "customer_id":              range(n),
            "credit_lines_outstanding": credit_lines,
            "loan_amt_outstanding":     loan_amt,
            "total_debt_outstanding":   debt,
            "income":                   income,
            "years_employed":           years_emp,
            "fico_score":               fico,
            "default":                  default_flag,
        })
        synthetic_df.to_csv(DATA_PATH, index=False)
        print(f"  Synthetic dataset saved as '{DATA_PATH}'  ({n:,} rows, "
              f"default rate={default_flag.mean():.2%})")

    # ── Train ─────────────────────────────────────────────────────────────
    calibrated_model, feature_list, X_test, y_test = train(DATA_PATH)

    # ── Single loan prediction examples ───────────────────────────────────
    print("\n" + "="*60)
    print("  EXPECTED LOSS PREDICTIONS — SAMPLE LOANS")
    print("="*60)

    sample_loans = [
        {
            "description": "Low-risk borrower (high income, long tenure, good payments)",
            "credit_lines_outstanding": 3,
            "loan_amt_outstanding": 10_000,
            "total_debt_outstanding": 8_000,
            "income": 120_000,
            "years_employed": 12,
            "fico_score": 780,          # prime borrower
        },
        {
            "description": "High-risk borrower (low income, short tenure, subprime FICO)",
            "credit_lines_outstanding": 10,
            "loan_amt_outstanding": 25_000,
            "total_debt_outstanding": 60_000,
            "income": 22_000,
            "years_employed": 0.5,
            "fico_score": 580,          # subprime
        },
        {
            "description": "Medium-risk borrower",
            "credit_lines_outstanding": 6,
            "loan_amt_outstanding": 15_000,
            "total_debt_outstanding": 30_000,
            "income": 55_000,
            "years_employed": 3,
            "fico_score": 660,          # near-prime
        },
    ]

    for loan in sample_loans:
        desc = loan.pop("description")
        result = predict_expected_loss(
            loan, model=calibrated_model, features=feature_list
        )
        print(f"\n  {desc}")
        print(f"    PD (Probability of Default) : {result['PD']:.2%}")
        print(f"    LGD (Loss Given Default)    : {result['LGD']:.0%}")
        print(f"    EAD (Exposure at Default)   : ${result['EAD']:>10,.2f}")
        print(f"    EL  (Expected Loss)         : ${result['EL']:>10,.2f}")
        loan["description"] = desc  # restore for readability

    print("\n  Done. Artefacts written: best_pd_model.pkl, "
          "calibration_curve.png, feature_importance.png")