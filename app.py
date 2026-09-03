# ============================================================
# CUSTOMER LOAN DECISION AI
# Streamlit + PyTorch
# ============================================================

import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib


# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Loan Decision AI",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Customer Loan Decision AI")

st.write(
    "Predict loan approval probability and estimated loan "
    "amount using PyTorch Artificial Neural Networks."
)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

st.caption(f"Running on: {device}")


# ============================================================
# 3. FEATURE COLUMNS
#
# These MUST match Cust_loan.ipynb exactly.
# ============================================================

CLASSIFICATION_COLUMNS = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value"
]


REGRESSION_COLUMNS = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value"
]


# ============================================================
# 4. CLASSIFICATION ANN
#
# EXACT architecture from Cust_loan.ipynb:
#
# input → 64 → 32 → 16 → 1
# ============================================================

class LoanClassifierANN(nn.Module):

    def __init__(self, input_features):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_features, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# 5. REGRESSION ANN
#
# EXACT architecture from Cust_loan.ipynb:
#
# input → 64 → 32 → 16 → 1
# ============================================================

class LoanRegressorANN(nn.Module):

    def __init__(self, input_features):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_features, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# 6. LOAD SCALERS
# ============================================================

@st.cache_resource
def load_scalers():

    classification_scaler = joblib.load(
        "classification_scaler.pkl"
    )

    regression_feature_scaler = joblib.load(
        "regression_feature_scaler.pkl"
    )

    regression_target_scaler = joblib.load(
        "regression_target_scaler.pkl"
    )

    return (
        classification_scaler,
        regression_feature_scaler,
        regression_target_scaler
    )


try:

    (
        classification_scaler,
        regression_feature_scaler,
        regression_target_scaler
    ) = load_scalers()

except Exception as e:

    st.error("❌ Could not load scaler files.")

    st.code(
        """
Required files:

classification_scaler.pkl
regression_feature_scaler.pkl
regression_target_scaler.pkl
        """
    )

    st.exception(e)

    st.stop()


# ============================================================
# 7. LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    model_cls = LoanClassifierANN(
        input_features=len(CLASSIFICATION_COLUMNS)
    ).to(device)


    model_cls.load_state_dict(
        torch.load(
            "loan_classifier.pth",
            map_location=device
        )
    )


    # --------------------------------------------------------
    # Regression
    # --------------------------------------------------------

    model_reg = LoanRegressorANN(
        input_features=len(REGRESSION_COLUMNS)
    ).to(device)


    model_reg.load_state_dict(
        torch.load(
            "loan_regressor.pth",
            map_location=device
        )
    )


    model_cls.eval()
    model_reg.eval()


    return model_cls, model_reg


try:

    model_cls, model_reg = load_models()

except Exception as e:

    st.error("❌ Could not load PyTorch model files.")

    st.code(
        """
Required files:

loan_classifier.pth
loan_regressor.pth
        """
    )

    st.exception(e)

    st.stop()


# ============================================================
# 8. CLASSIFICATION PREDICTION
# ============================================================

def predict_loan_approval(customer):

    model_cls.eval()


    # Make sure columns are exactly the same
    # order as training data

    customer_cls = customer[
        CLASSIFICATION_COLUMNS
    ].copy()


    # Apply the SAME scaler used during training

    X_scaled = classification_scaler.transform(
        customer_cls
    )


    # Convert to PyTorch tensor

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    ).to(device)


    # Prediction

    with torch.no_grad():

        logits = model_cls(X_tensor)

        probability = torch.sigmoid(
            logits
        ).item()


    prediction = (
        1 if probability >= 0.5 else 0
    )


    return prediction, probability


# ============================================================
# 9. REGRESSION PREDICTION
# ============================================================

def predict_loan_amount(customer):

    model_reg.eval()


    # Regression model does NOT use loan_amount.
    # This matches Cust_loan.ipynb.

    regression_input = customer[
        REGRESSION_COLUMNS
    ].copy()


    # Apply regression feature scaler

    X_scaled = regression_feature_scaler.transform(
        regression_input
    )


    # Convert to tensor

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    ).to(device)


    # Prediction

    with torch.no_grad():

        prediction_scaled = model_reg(
            X_tensor
        )


    # Move prediction to CPU

    prediction_scaled = (
        prediction_scaled
        .cpu()
        .numpy()
    )


    # Convert scaled prediction back
    # to original loan amount

    prediction = regression_target_scaler.inverse_transform(
        prediction_scaled
    )


    return prediction.item()


# ============================================================
# 10. COMPLETE LOAN DECISION SYSTEM
# ============================================================

def loan_decision_system(customer):

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    approval, probability = predict_loan_approval(
        customer
    )


    # --------------------------------------------------------
    # If rejected
    # --------------------------------------------------------

    if approval == 0:

        return {
            "decision": "REJECTED",
            "probability": probability,
            "loan_amount": None
        }


    # --------------------------------------------------------
    # If approved → regression
    # --------------------------------------------------------

    predicted_amount = predict_loan_amount(
        customer
    )


    return {
        "decision": "APPROVED",
        "probability": probability,
        "loan_amount": predicted_amount
    }


# ============================================================
# 11. CUSTOMER INPUT FORM
# ============================================================

st.subheader("👤 Customer Information")


with st.form("loan_form"):

    # --------------------------------------------------------
    # Personal Information
    # --------------------------------------------------------

    no_of_dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )


    education = st.selectbox(
        "Education",
        [
            "Graduate",
            "Not Graduate"
        ]
    )


    self_employed = st.selectbox(
        "Self Employed",
        [
            "Yes",
            "No"
        ]
    )


    # --------------------------------------------------------
    # Financial Information
    # --------------------------------------------------------

    income_annum = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=5000000.0,
        step=100000.0
    )


    loan_amount = st.number_input(
        "Requested Loan Amount",
        min_value=0.0,
        value=10000000.0,
        step=100000.0
    )


    loan_term = st.number_input(
        "Loan Term (Years)",
        min_value=1,
        max_value=50,
        value=10,
        step=1
    )


    cibil_score = st.number_input(
        "CIBIL Score",
        min_value=0.0,
        max_value=1000.0,
        value=750.0,
        step=1.0
    )


    # --------------------------------------------------------
    # Assets
    # --------------------------------------------------------

    residential_assets_value = st.number_input(
        "Residential Assets Value",
        min_value=0.0,
        value=5000000.0,
        step=100000.0
    )


    commercial_assets_value = st.number_input(
        "Commercial Assets Value",
        min_value=0.0,
        value=0.0,
        step=100000.0
    )


    luxury_assets_value = st.number_input(
        "Luxury Assets Value",
        min_value=0.0,
        value=0.0,
        step=100000.0
    )


    bank_asset_value = st.number_input(
        "Bank Asset Value",
        min_value=0.0,
        value=1000000.0,
        step=100000.0
    )


    # --------------------------------------------------------
    # Submit
    # --------------------------------------------------------

    submitted = st.form_submit_button(
        "🔍 Check Loan Eligibility"
    )


# ============================================================
# 12. PREDICTION
# ============================================================

if submitted:

    # --------------------------------------------------------
    # Convert Education
    # Same mapping used in notebook
    # --------------------------------------------------------

    education_value = (
        1 if education == "Graduate"
        else 0
    )


    # --------------------------------------------------------
    # Convert Self Employed
    # Same mapping used in notebook
    # --------------------------------------------------------

    self_employed_value = (
        1 if self_employed == "Yes"
        else 0
    )


    # --------------------------------------------------------
    # Create customer DataFrame
    # --------------------------------------------------------

    customer_data = pd.DataFrame([{

        "no_of_dependents": no_of_dependents,

        "education": education_value,

        "self_employed": self_employed_value,

        "income_annum": income_annum,

        "loan_amount": loan_amount,

        "loan_term": loan_term,

        "cibil_score": cibil_score,

        "residential_assets_value":
            residential_assets_value,

        "commercial_assets_value":
            commercial_assets_value,

        "luxury_assets_value":
            luxury_assets_value,

        "bank_asset_value":
            bank_asset_value

    }])


    try:

        # ----------------------------------------------------
        # Run model
        # ----------------------------------------------------

        result = loan_decision_system(
            customer_data
        )


        probability = (
            result["probability"] * 100
        )


        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        st.divider()

        st.subheader("📊 Prediction Result")


        st.metric(
            "Approval Probability",
            f"{probability:.2f}%"
        )


        # ----------------------------------------------------
        # APPROVED
        # ----------------------------------------------------

        if result["decision"] == "APPROVED":

            st.success(
                "✅ Loan Approved"
            )


            st.metric(
                "Predicted Loan Amount",
                f"₹{result['loan_amount']:,.2f}"
            )


            st.info(
                "The Classification ANN predicts approval. "
                "The Regression ANN then predicts the loan amount."
            )


        # ----------------------------------------------------
        # REJECTED
        # ----------------------------------------------------

        else:

            st.error(
                "❌ Loan Rejected"
            )


            st.info(
                "The Classification ANN predicts that "
                "this application does not meet the "
                "model's approval threshold."
            )


        # ----------------------------------------------------
        # Customer information
        # ----------------------------------------------------

        with st.expander(
            "👤 View Customer Information"
        ):

            st.dataframe(
                customer_data,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(e)


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with PyTorch + Streamlit | "
    "Customer Loan Decision & Amount Prediction"
)