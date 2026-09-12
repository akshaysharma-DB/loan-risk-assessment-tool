import pandas as pd
import joblib

from pathlib import Path


# =========================================================
# LOAD TRAINED MODEL AND PREPROCESSING OBJECTS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(
    MODEL_DIR / "logistic_model_v3.pkl"
)

scaler = joblib.load(
    MODEL_DIR / "scaler_v3.pkl"
)

encoder = joblib.load(
    MODEL_DIR / "encoder_v3.pkl"
)

feature_columns = joblib.load(
    MODEL_DIR / "features_v3.pkl"
)


# =========================================================
# MODEL INPUT FEATURES
# =========================================================

NUMERIC_FEATURES = [
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


CATEGORICAL_FEATURES = [
    "purpose",
    "home_ownership"
]


# =========================================================
# DISPLAY NAMES
# =========================================================

DISPLAY_NAMES = {

    "loan_amnt": "Loan Amount",
    "funded_amnt": "Funded Amount",
    "term": "Loan Term",
    "int_rate": "Interest Rate",
    "annual_inc": "Annual Income",
    "emp_length": "Employment Length",
    "dti": "Debt-to-Income Ratio",
    "installment": "Monthly Installment",
    "credit_history_years": "Credit History",
    "delinq_2yrs": "Recent Delinquencies",
    "inq_last_6mths": "Recent Credit Inquiries",
    "open_acc": "Open Credit Accounts",
    "total_acc": "Total Credit Accounts",
    "revol_bal": "Revolving Balance",
    "revol_util": "Revolving Utilization",
    "mort_acc": "Mortgage Accounts",
    "tot_cur_bal": "Total Current Balance",
    "pub_rec": "Public Records",
    "funded_amnt_inv": "Investor-Funded Amount"
}


# =========================================================
# VALUE FORMATTING
# =========================================================

def format_value(feature, value):

    if feature in [
        "loan_amnt",
        "funded_amnt",
        "funded_amnt_inv",
        "annual_inc",
        "installment",
        "revol_bal",
        "tot_cur_bal"
    ]:

        return f"${value:,.0f}"


    if feature in [
        "int_rate",
        "dti",
        "revol_util"
    ]:

        return f"{value:.1f}%"


    if feature == "term":

        return f"{int(value)} months"


    if feature == "emp_length":

        return f"{value:.0f} years"


    if feature == "credit_history_years":

        return f"{value:.1f} years"


    if feature in [
        "delinq_2yrs",
        "inq_last_6mths",
        "open_acc",
        "total_acc",
        "mort_acc",
        "pub_rec"
    ]:

        return f"{value:.0f}"


    return str(value)


# =========================================================
# RISK EXPLANATIONS
# =========================================================

def get_risk_explanation(feature, value):

    explanations = {

        "loan_amnt":
            "A larger loan amount increases the borrower's overall repayment obligation and can increase the financial impact if repayment becomes difficult.",

        "funded_amnt":
            "A larger funded amount means the borrower is taking on a greater principal obligation.",

        "funded_amnt_inv":
            "A higher investor-funded amount represents a larger amount of credit being extended to the borrower.",

        "term":
            "A longer repayment term keeps the borrower in debt for longer and increases the period over which repayment risk is exposed.",

        "int_rate":
            "A higher interest rate increases the cost of borrowing and can result in a larger repayment burden for the borrower.",

        "annual_inc":
            "Lower income provides less financial capacity to absorb existing and new debt obligations.",

        "emp_length":
            "Shorter employment history may indicate less established income stability.",

        "dti":
            "A higher debt-to-income ratio means a larger share of the borrower's income is already committed to debt obligations, leaving less capacity for additional debt.",

        "installment":
            "A higher monthly installment creates a larger recurring repayment obligation for the borrower.",

        "credit_history_years":
            "A shorter credit history provides less historical information about the borrower's long-term credit behavior.",

        "delinq_2yrs":
            "Recent delinquencies indicate missed or late credit payments and can signal difficulty meeting existing credit obligations.",

        "inq_last_6mths":
            "Multiple recent credit inquiries may indicate that the borrower has recently sought additional credit, which can be a sign of increased borrowing activity.",

        "open_acc":
            "A larger number of currently open accounts can indicate that the borrower already has multiple active credit obligations.",

        "total_acc":
            "A large number of credit accounts can indicate greater exposure to multiple credit obligations.",

        "revol_bal":
            "A higher revolving balance represents more outstanding revolving debt that the borrower must repay.",

        "revol_util":
            "High revolving credit utilization means the borrower is using a large portion of available revolving credit, which can indicate greater credit stress.",

        "mort_acc":
            "Mortgage accounts represent additional long-term financial obligations that may affect the borrower's overall debt exposure.",

        "tot_cur_bal":
            "A higher outstanding balance indicates greater existing debt obligations, which may reduce the borrower's capacity to take on additional debt.",

        "pub_rec":
            "Public records on a credit profile can indicate previous serious credit-related issues and may increase concern about repayment risk."
    }

    return explanations.get(
        feature,
        "This applicant characteristic is associated with a higher estimated default risk in the model."
    )


# =========================================================
# PROTECTIVE EXPLANATIONS
# =========================================================

def get_protective_explanation(feature, value):

    explanations = {

        "loan_amnt":
            "A smaller loan amount results in a lower overall repayment obligation.",

        "funded_amnt":
            "A smaller funded amount results in a lower principal obligation.",

        "int_rate":
            "A lower interest rate reduces the cost of borrowing and the associated repayment burden.",

        "annual_inc":
            "Higher income can provide greater capacity to meet debt obligations.",

        "emp_length":
            "Longer employment history can indicate more established income stability.",

        "dti":
            "A lower debt-to-income ratio indicates that less of the borrower's income is already committed to debt.",

        "installment":
            "A lower monthly installment creates a smaller recurring repayment obligation.",

        "credit_history_years":
            "A longer credit history provides more historical information about the borrower's credit behavior.",

        "delinq_2yrs":
            "Fewer recent delinquencies indicate a stronger recent payment history.",

        "inq_last_6mths":
            "Fewer recent credit inquiries indicate less recent demand for additional credit.",

        "open_acc":
            "Fewer active credit accounts may indicate lower exposure to multiple current credit obligations.",

        "total_acc":
            "A longer-established credit profile can provide more historical information about credit behavior.",

        "revol_bal":
            "A lower revolving balance represents less outstanding revolving debt.",

        "revol_util":
            "Lower revolving utilization indicates that more available revolving credit remains unused.",

        "mort_acc":
            "A lower number of mortgage accounts indicates fewer mortgage-related obligations.",

        "tot_cur_bal":
            "A lower outstanding balance indicates lower existing debt obligations.",

        "pub_rec":
            "No public records avoids an additional credit-related concern in the applicant's profile."
    }

    return explanations.get(
        feature,
        "This applicant characteristic is associated with lower estimated default risk in the model."
    )


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_loan(applicant):

    data = pd.DataFrame([applicant])


    # -----------------------------------------------------
    # SEPARATE INPUTS
    # -----------------------------------------------------

    numeric_data = data[
        NUMERIC_FEATURES
    ].copy()

    categorical_data = data[
        CATEGORICAL_FEATURES
    ].copy()


    # -----------------------------------------------------
    # ENCODE CATEGORICAL FEATURES
    # -----------------------------------------------------

    encoded_data = encoder.transform(
        categorical_data
    )

    encoded_columns = encoder.get_feature_names_out(
        CATEGORICAL_FEATURES
    )

    encoded_data = pd.DataFrame(
        encoded_data,
        columns=encoded_columns
    )


    # -----------------------------------------------------
    # CREATE MODEL INPUT
    # -----------------------------------------------------

    model_input = pd.concat(
        [
            numeric_data,
            encoded_data
        ],
        axis=1
    )


    model_input = model_input[
        feature_columns
    ]


    # -----------------------------------------------------
    # SCALE
    # -----------------------------------------------------

    model_input_scaled = scaler.transform(
        model_input
    )


    # -----------------------------------------------------
    # DEFAULT PROBABILITY
    # -----------------------------------------------------

    probability = model.predict_proba(
        model_input_scaled
    )[0][1]


    # -----------------------------------------------------
    # DECISION THRESHOLD
    # -----------------------------------------------------

    threshold = 0.30


    if probability >= threshold:

        risk_level = "Higher Risk"
        recommendation = "Further Review"

    else:

        risk_level = "Lower Risk"
        recommendation = "Standard Review"


    # =====================================================
    # APPLICANT-SPECIFIC CONTRIBUTIONS
    # =====================================================

    coefficients = model.coef_[0]

    contributions = (
        model_input_scaled[0] * coefficients
    )


    contribution_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "contribution": contributions
        }
    )


    contribution_df["abs_contribution"] = (
        contribution_df["contribution"].abs()
    )


    # Ignore extremely small contributions.

    contribution_df = contribution_df[
        contribution_df["abs_contribution"] >= 0.02
    ].sort_values(
        "abs_contribution",
        ascending=False
    )


    # =====================================================
    # RISK & PROTECTIVE FACTORS
    # =====================================================

    risk_factors = []

    protective_factors = []


    for _, row in contribution_df.iterrows():

        feature = row["feature"]

        contribution = row["contribution"]

        value = model_input.iloc[0][feature]


        # -------------------------------------------------
        # HANDLE ONE-HOT FEATURES
        # -------------------------------------------------

        if feature.startswith("purpose_"):

            if value != 1:
                continue

            readable_name = (
                "Loan Purpose: "
                + feature.replace(
                    "purpose_", ""
                ).replace(
                    "_", " "
                ).title()
            )

            display_value = readable_name.replace(
                "Loan Purpose: ", ""
            )


            factor = {
                "feature": readable_name,
                "value": display_value,
                "display_value": display_value,
                "contribution": contribution
            }


        elif feature.startswith("home_ownership_"):

            if value != 1:
                continue

            readable_name = (
                "Home Ownership: "
                + feature.replace(
                    "home_ownership_", ""
                ).title()
            )

            display_value = readable_name.replace(
                "Home Ownership: ", ""
            )


            factor = {
                "feature": readable_name,
                "value": display_value,
                "display_value": display_value,
                "contribution": contribution
            }


        else:

            readable_name = DISPLAY_NAMES.get(
                feature,
                feature.replace(
                    "_", " "
                ).title()
            )

            display_value = format_value(
                feature,
                value
            )


            factor = {
                "feature": readable_name,
                "value": value,
                "display_value": display_value,
                "contribution": contribution
            }


        # -------------------------------------------------
        # HIGHER-RISK CONTRIBUTION
        # -------------------------------------------------

        if contribution > 0:

            if not (
                feature.startswith("purpose_")
                or feature.startswith("home_ownership_")
            ):

                factor["explanation"] = (
                    get_risk_explanation(
                        feature,
                        value
                    )
                )

            else:

                factor["explanation"] = (
                    "This loan characteristic is associated "
                    "with higher estimated default risk in "
                    "the model."
                )


            risk_factors.append(
                factor
            )


        # -------------------------------------------------
        # LOWER-RISK CONTRIBUTION
        # -------------------------------------------------

        elif contribution < 0:

            if not (
                feature.startswith("purpose_")
                or feature.startswith("home_ownership_")
            ):

                factor["explanation"] = (
                    get_protective_explanation(
                        feature,
                        value
                    )
                )

            else:

                factor["explanation"] = (
                    "This loan characteristic is associated "
                    "with lower estimated default risk in "
                    "the model."
                )


            protective_factors.append(
                factor
            )


    # =====================================================
    # RETURN RESULTS
    # =====================================================

    return {

        "default_probability": probability,

        "risk_level": risk_level,

        "threshold": threshold,

        "recommendation": recommendation,

        "risk_factors": risk_factors[:5],

        "protective_factors": protective_factors[:5]
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    sample_applicant = {

        "loan_amnt": 15000,

        "funded_amnt": 15000,

        "term": 36,

        "int_rate": 12.5,

        "annual_inc": 60000,

        "emp_length": 5,

        "dti": 18.5,

        "installment": 501.25,

        "purpose": "debt_consolidation",

        "credit_history_years": 12,

        "home_ownership": "RENT",

        "delinq_2yrs": 0,

        "inq_last_6mths": 1,

        "open_acc": 8,

        "total_acc": 20,

        "revol_bal": 7500,

        "revol_util": 35.0,

        "mort_acc": 0,

        "tot_cur_bal": 18000,

        "pub_rec": 0
    }


    result = predict_loan(
        sample_applicant
    )


    print("\nLoan Risk Assessment")
    print("--------------------")

    print(
        f"Default Probability: "
        f"{result['default_probability']:.2%}"
    )

    print(
        f"Risk Level: "
        f"{result['risk_level']}"
    )

    print(
        f"Decision Threshold: "
        f"{result['threshold']:.0%}"
    )

    print(
        f"Recommendation: "
        f"{result['recommendation']}"
    )


    print("\nRisk Factors:")

    for factor in result["risk_factors"]:

        print(
            f"\n- {factor['feature']} "
            f"({factor['display_value']})"
        )

        print(
            f"  {factor['explanation']}"
        )


    print("\nFactors Reducing Estimated Risk:")

    for factor in result["protective_factors"]:

        print(
            f"\n- {factor['feature']} "
            f"({factor['display_value']})"
        )

        print(
            f"  {factor['explanation']}"
        )