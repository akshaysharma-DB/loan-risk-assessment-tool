import pandas as pd
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from Paths import CLEANED_DIR


# Load the same train/test split used in the previous models.
# Keeping the split unchanged makes the model comparison fair.
X_train = pd.read_csv(CLEANED_DIR / "X_train.csv")
X_test = pd.read_csv(CLEANED_DIR / "X_test.csv")

y_train = pd.read_csv(CLEANED_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(CLEANED_DIR / "y_test.csv").squeeze()

print(X_train.shape, X_test.shape)


# These are the fields that a lender can realistically provide
# when evaluating a new loan application.
#
# Grade and sub-grade are deliberately not included because
# they already represent an existing risk classification.
numeric_features = [
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "annual_inc",
    "emp_length",
    "dti",
    "installment",
    "credit_history_years",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "total_acc",
    "revol_bal",
    "revol_util",
    "mort_acc",
    "tot_cur_bal",
    "pub_rec"
]

categorical_features = [
    "purpose",
    "home_ownership"
]


# The original training data contains one-hot encoded versions
# of purpose and home ownership, so reconstruct the categorical
# values from those columns before training the application model.
def reconstruct_category(df, prefix, categories):
    result = pd.Series("Unknown", index=df.index)

    for category in categories:
        column = f"{prefix}_{category}"

        if column in df.columns:
            mask = df[column] == 1
            result.loc[mask] = category

    return result


purpose_categories = [
    "credit_card",
    "debt_consolidation",
    "educational",
    "home_improvement",
    "house",
    "major_purchase",
    "medical",
    "moving",
    "other",
    "renewable_energy",
    "small_business",
    "vacation",
    "wedding"
]

home_ownership_categories = [
    "MORTGAGE",
    "NONE",
    "OTHER",
    "OWN",
    "RENT"
]


X_train_app = X_train[numeric_features].copy()
X_test_app = X_test[numeric_features].copy()

X_train_app["purpose"] = reconstruct_category(
    X_train,
    "purpose",
    purpose_categories
)

X_test_app["purpose"] = reconstruct_category(
    X_test,
    "purpose",
    purpose_categories
)

X_train_app["home_ownership"] = reconstruct_category(
    X_train,
    "home_ownership",
    home_ownership_categories
)

X_test_app["home_ownership"] = reconstruct_category(
    X_test,
    "home_ownership",
    home_ownership_categories
)


# Encode the two categorical fields into numerical columns.
# The encoder is fitted only on the training data and reused
# later by the prediction application.
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

X_train_cat = encoder.fit_transform(
    X_train_app[categorical_features]
)

X_test_cat = encoder.transform(
    X_test_app[categorical_features]
)


encoded_columns = encoder.get_feature_names_out(
    categorical_features
)


X_train_cat = pd.DataFrame(
    X_train_cat,
    columns=encoded_columns,
    index=X_train_app.index
)

X_test_cat = pd.DataFrame(
    X_test_cat,
    columns=encoded_columns,
    index=X_test_app.index
)


# Combine numerical and encoded categorical variables.
X_train_final = pd.concat(
    [
        X_train_app[numeric_features],
        X_train_cat
    ],
    axis=1
)

X_test_final = pd.concat(
    [
        X_test_app[numeric_features],
        X_test_cat
    ],
    axis=1
)


print("Application model features:", X_train_final.shape[1])


# Scale the numerical model inputs.
# The scaler is saved separately so the exact same transformation
# can be applied when a lender evaluates a new applicant.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)


# Balanced class weights give additional importance to defaulted
# loans because defaults represent a smaller portion of the data.
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(
    X_train_scaled,
    y_train
)


# Save the model and preprocessing objects.
# These files will later be used by the Streamlit application.
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(
    model,
    MODEL_DIR / "logistic_model_v3.pkl"
)

joblib.dump(
    scaler,
    MODEL_DIR / "scaler_v3.pkl"
)

joblib.dump(
    encoder,
    MODEL_DIR / "encoder_v3.pkl"
)

joblib.dump(
    X_train_final.columns.tolist(),
    MODEL_DIR / "features_v3.pkl"
)

print("Saved V3 application model and preprocessing files.")


# Generate predictions using the standard 0.50 threshold.
y_pred = model.predict(X_test_scaled)

y_prob = model.predict_proba(
    X_test_scaled
)[:, 1]


print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ROC-AUC measures how well the model separates defaulted
# and non-defaulted loans across different thresholds.
roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("ROC-AUC:", round(roc_auc, 4))


# Show the features that have the strongest relationship with
# predicted default risk.
coefficients = pd.DataFrame({
    "feature": X_train_final.columns,
    "coefficient": model.coef_[0]
})

coefficients["abs_coefficient"] = (
    coefficients["coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "abs_coefficient",
    ascending=False
)

print("\nTop application-model features:")
print(
    coefficients.head(20).to_string(index=False)
)