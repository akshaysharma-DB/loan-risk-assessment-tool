import streamlit as st

from src.predict import predict_loan


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Loan Risk Assessment Tool",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #666;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .section-text {
        color: #666;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">Loan Default Risk Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Estimate the probability of loan default using borrower '
    'and credit information.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAN & APPLICANT DETAILS
# =========================================================

st.header("Loan & Applicant Details")

col1, col2 = st.columns(2)


with col1:

    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=15000.0,
        step=500.0
    )

    funded_amnt = st.number_input(
        "Funded Amount",
        min_value=0.0,
        value=15000.0,
        step=500.0
    )

    term = st.selectbox(
        "Loan Term",
        [36, 60],
        format_func=lambda x: f"{x} months"
    )

    int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=12.5,
        step=0.1
    )

    annual_inc = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=60000.0,
        step=1000.0
    )

    emp_length = st.number_input(
        "Employment Length (years)",
        min_value=0,
        max_value=10,
        value=5
    )


with col2:

    dti = st.number_input(
        "Debt-to-Income Ratio (%)",
        min_value=0.0,
        max_value=100.0,
        value=18.5,
        step=0.1
    )

    installment = st.number_input(
        "Monthly Installment",
        min_value=0.0,
        value=501.25,
        step=10.0
    )

    purpose = st.selectbox(
        "Loan Purpose",
        [
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
        ],
        format_func=lambda x: x.replace("_", " ").title()
    )

    home_ownership = st.selectbox(
        "Home Ownership",
        [
            "MORTGAGE",
            "NONE",
            "OTHER",
            "OWN",
            "RENT"
        ],
        format_func=lambda x: x.title()
    )

    credit_history_years = st.number_input(
        "Credit History Length (years)",
        min_value=0.0,
        value=12.0,
        step=0.5
    )


# =========================================================
# CREDIT PROFILE
# =========================================================

st.header("Credit Profile")

col1, col2, col3 = st.columns(3)


with col1:

    delinq_2yrs = st.number_input(
        "Delinquencies (last 2 years)",
        min_value=0,
        value=0,
        step=1
    )

    inq_last_6mths = st.number_input(
        "Credit Inquiries (last 6 months)",
        min_value=0,
        value=1,
        step=1
    )

    open_acc = st.number_input(
        "Open Credit Accounts",
        min_value=0,
        value=8,
        step=1
    )


with col2:

    total_acc = st.number_input(
        "Total Credit Accounts",
        min_value=0,
        value=20,
        step=1
    )

    revol_bal = st.number_input(
        "Revolving Balance",
        min_value=0.0,
        value=7500.0,
        step=500.0
    )

    revol_util = st.number_input(
        "Revolving Utilization (%)",
        min_value=0.0,
        max_value=100.0,
        value=35.0,
        step=1.0
    )


with col3:

    mort_acc = st.number_input(
        "Mortgage Accounts",
        min_value=0,
        value=0,
        step=1
    )

    tot_cur_bal = st.number_input(
        "Total Current Balance",
        min_value=0.0,
        value=18000.0,
        step=500.0
    )

    pub_rec = st.number_input(
        "Public Records",
        min_value=0,
        value=0,
        step=1
    )


# =========================================================
# ASSESS LOAN RISK
# =========================================================

st.divider()

if st.button(
    "Assess Loan Risk",
    type="primary",
    use_container_width=True
):

    applicant = {

        "loan_amnt": loan_amnt,
        "funded_amnt": funded_amnt,
        "term": term,
        "int_rate": int_rate,
        "annual_inc": annual_inc,
        "emp_length": emp_length,
        "dti": dti,
        "installment": installment,
        "purpose": purpose,
        "credit_history_years": credit_history_years,
        "home_ownership": home_ownership,

        "delinq_2yrs": delinq_2yrs,
        "inq_last_6mths": inq_last_6mths,
        "open_acc": open_acc,
        "total_acc": total_acc,
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "mort_acc": mort_acc,
        "tot_cur_bal": tot_cur_bal,
        "pub_rec": pub_rec
    }


    try:

        # =================================================
        # RUN MODEL
        # =================================================

        result = predict_loan(applicant)

        probability = result["default_probability"]

        risk_level = result["risk_level"]

        recommendation = result["recommendation"]

        threshold = result["threshold"]

        risk_factors = result["risk_factors"]

        protective_factors = result["protective_factors"]


        # =================================================
        # RISK ASSESSMENT
        # =================================================

        st.header("Risk Assessment")

        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.metric(
                "Estimated Default Probability",
                f"{probability:.2%}"
            )

            st.progress(
                min(probability, 1.0),
                text=f"Estimated risk: {probability:.2%}"
            )


        with result_col2:

            if risk_level == "Higher Risk":

                st.error(
                    f"⚠️ {risk_level}"
                )

            else:

                st.success(
                    f"✓ {risk_level}"
                )

            st.subheader(
                recommendation
            )


        # =================================================
        # DECISION EXPLANATION
        # =================================================

        st.divider()

        st.subheader("Decision Explanation")


        if probability >= threshold:

            st.warning(
                f"The estimated probability of default is "
                f"**{probability:.2%}**, which is above the "
                f"configured **{threshold:.0%} review threshold**."
            )

            st.write(
                "The application is therefore flagged for "
                "further review rather than being automatically "
                "approved."
            )

        else:

            st.success(
                f"The estimated probability of default is "
                f"**{probability:.2%}**, which is below the "
                f"configured **{threshold:.0%} review threshold**."
            )

            st.write(
                "The application falls below the model's "
                "configured risk threshold and can proceed "
                "to standard review."
            )


        # =================================================
        # RISK FACTORS
        # =================================================

        st.divider()

        st.subheader("Risk Factors")

        st.markdown(
            '<div class="section-text">'
            'Applicant characteristics that are contributing '
            'toward a higher estimated default risk.'
            '</div>',
            unsafe_allow_html=True
        )


        if risk_factors:

            for factor in risk_factors:

                st.warning(
                    f"**{factor['feature']} — "
                    f"{factor['display_value']}**\n\n"
                    f"{factor['explanation']}"
                )

        else:

            st.success(
                "No major applicant-specific factors were "
                "identified as increasing the estimated risk."
            )


        # =================================================
        # FACTORS REDUCING RISK
        # =================================================

        if protective_factors:

            st.divider()

            st.subheader(
                "Factors Reducing Estimated Risk"
            )

            st.markdown(
                '<div class="section-text">'
                'Applicant characteristics that are contributing '
                'toward a lower estimated default risk.'
                '</div>',
                unsafe_allow_html=True
            )


            for factor in protective_factors:

                st.success(
                    f"**{factor['feature']} — "
                    f"{factor['display_value']}**\n\n"
                    f"{factor['explanation']}"
                )


        # =================================================
        # APPLICATION SUMMARY
        # =================================================

        st.divider()

        st.subheader("Application Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)


        with summary_col1:

            st.metric(
                "Loan Amount",
                f"${loan_amnt:,.0f}"
            )

            st.metric(
                "Annual Income",
                f"${annual_inc:,.0f}"
            )


        with summary_col2:

            st.metric(
                "Interest Rate",
                f"{int_rate:.2f}%"
            )

            st.metric(
                "Debt-to-Income",
                f"{dti:.2f}%"
            )


        with summary_col3:

            st.metric(
                "Credit History",
                f"{credit_history_years:.1f} years"
            )

            st.metric(
                "Loan Term",
                f"{term} months"
            )


        # =================================================
        # MODEL INFORMATION
        # =================================================

        st.divider()

        st.subheader("Model Information")

        info_col1, info_col2, info_col3 = st.columns(3)


        with info_col1:

            st.metric(
                "Decision Threshold",
                f"{threshold:.0%}"
            )


        with info_col2:

            st.metric(
                "Model",
                "Logistic Regression"
            )


        with info_col3:

            st.metric(
                "Risk Output",
                "Default Probability"
            )


        # =================================================
        # DISCLAIMER
        # =================================================

        st.divider()

        st.caption(
            "This application is a decision-support tool. "
            "The model's output should be considered alongside "
            "the lender's underwriting policies and other "
            "relevant information. Model explanations describe "
            "features contributing to the prediction and should "
            "not be interpreted as proof of causation."
        )


    except Exception as e:

        st.error(
            "Unable to generate the risk assessment."
        )

        st.exception(e)
