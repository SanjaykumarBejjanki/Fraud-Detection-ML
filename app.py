import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fraud Detection AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: #f5f7fb;
    }

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Header */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #667085;
        margin-bottom: 25px;
    }

    /* Cards */
    .info-card {
        background: white;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #e4e7ec;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-bottom: 15px;
    }

    .card-title {
        font-size: 15px;
        color: #667085;
        margin-bottom: 5px;
    }

    .card-value {
        font-size: 28px;
        font-weight: 700;
    }

    /* Result cards */
    .fraud-result {
        background: #fff1f0;
        border: 1px solid #ffccc7;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
    }

    .safe-result {
        background: #f0fff4;
        border: 1px solid #b7ebc6;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
    }

    .result-title {
        font-size: 30px;
        font-weight: 800;
    }

    .result-description {
        font-size: 16px;
        color: #667085;
    }

    /* Section headings */
    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 45px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #e4e7ec;
    }

    /* Input labels */
    label {
        font-weight: 600 !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #98a2b3;
        padding: 25px;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler file not found: {SCALER_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


try:

    model, scaler = load_model()

except Exception as e:

    st.error("❌ Unable to load the machine learning model.")

    st.code(str(e))

    st.info(
        "Make sure fraud_model.pkl and scaler.pkl exist inside the models folder."
    )

    st.stop()


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount"
]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 💳 Fraud AI")

    st.markdown("---")

    st.markdown("### 🤖 Model Information")

    st.write("**Algorithm:** Logistic Regression")
    st.write("**Dataset:** Credit Card Transactions")
    st.write("**Features:** 30")
    st.write("**Classes:** 2")

    st.markdown("---")

    st.markdown("### 📊 Model Performance")

    st.metric("ROC-AUC", "97.10%")
    st.metric("Fraud Recall", "91.84%")
    st.metric("Fraud Precision", "5.79%")

    st.markdown("---")

    st.info(
        "This system uses machine learning to identify potentially "
        "fraudulent credit card transactions."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">💳 Credit Card Fraud Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered transaction risk analysis using machine learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# DASHBOARD METRICS
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:

    st.markdown(
        """
        <div class="info-card">
            <div class="card-title">Model</div>
            <div class="card-value">Logistic Regression</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric2:

    st.markdown(
        """
        <div class="info-card">
            <div class="card-title">ROC-AUC</div>
            <div class="card-value">97.10%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric3:

    st.markdown(
        """
        <div class="info-card">
            <div class="card-title">Fraud Recall</div>
            <div class="card-value">91.84%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric4:

    st.markdown(
        """
        <div class="info-card">
            <div class="card-title">Features</div>
            <div class="card-value">30</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TRANSACTION INPUT
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Transaction Analysis</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the transaction characteristics below and click "
    "**Analyze Transaction**."
)


# ============================================================
# INPUT COLUMNS
# ============================================================

col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# COLUMN 1
# ------------------------------------------------------------

with col1:

    st.markdown("### Transaction Information")

    time = st.number_input(
        "Transaction Time",
        min_value=0.0,
        value=0.0,
        step=1.0,
        help="Time elapsed since the first transaction in the dataset."
    )

    amount = st.number_input(
        "💰 Transaction Amount",
        min_value=0.0,
        value=100.0,
        step=1.0,
        help="Transaction amount."
    )


# ------------------------------------------------------------
# COLUMN 2
# ------------------------------------------------------------

with col2:

    st.markdown("### Features V1 – V9")

    v1 = st.number_input("V1", value=0.0)
    v2 = st.number_input("V2", value=0.0)
    v3 = st.number_input("V3", value=0.0)
    v4 = st.number_input("V4", value=0.0)
    v5 = st.number_input("V5", value=0.0)
    v6 = st.number_input("V6", value=0.0)
    v7 = st.number_input("V7", value=0.0)
    v8 = st.number_input("V8", value=0.0)
    v9 = st.number_input("V9", value=0.0)


# ------------------------------------------------------------
# COLUMN 3
# ------------------------------------------------------------

with col3:

    st.markdown("### Features V10 – V18")

    v10 = st.number_input("V10", value=0.0)
    v11 = st.number_input("V11", value=0.0)
    v12 = st.number_input("V12", value=0.0)
    v13 = st.number_input("V13", value=0.0)
    v14 = st.number_input("V14", value=0.0)
    v15 = st.number_input("V15", value=0.0)
    v16 = st.number_input("V16", value=0.0)
    v17 = st.number_input("V17", value=0.0)
    v18 = st.number_input("V18", value=0.0)


# ============================================================
# REMAINING FEATURES
# ============================================================

with st.expander("⚙️ Advanced Transaction Features — V19 to V28"):

    advanced_col1, advanced_col2 = st.columns(2)

    with advanced_col1:

        v19 = st.number_input("V19", value=0.0)
        v20 = st.number_input("V20", value=0.0)
        v21 = st.number_input("V21", value=0.0)
        v22 = st.number_input("V22", value=0.0)
        v23 = st.number_input("V23", value=0.0)

    with advanced_col2:

        v24 = st.number_input("V24", value=0.0)
        v25 = st.number_input("V25", value=0.0)
        v26 = st.number_input("V26", value=0.0)
        v27 = st.number_input("V27", value=0.0)
        v28 = st.number_input("V28", value=0.0)


st.divider()


# ============================================================
# BUTTONS
# ============================================================

button_col1, button_col2 = st.columns([3, 1])

with button_col1:

    analyze = st.button(
        "🚨 Analyze Transaction",
        use_container_width=True,
        type="primary"
    )

with button_col2:

    reset = st.button(
        "🔄 Reset",
        use_container_width=True
    )

if reset:

    st.rerun()


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # We use a DataFrame instead of numpy array.
    #
    # This preserves the feature names used during scaler
    # training and removes:
    #
    # "X does not have valid feature names"
    #
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [[
            time,
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
            v8,
            v9,
            v10,
            v11,
            v12,
            v13,
            v14,
            v15,
            v16,
            v17,
            v18,
            v19,
            v20,
            v21,
            v22,
            v23,
            v24,
            v25,
            v26,
            v27,
            v28,
            amount
        ]],
        columns=FEATURE_NAMES
    )


    # --------------------------------------------------------
    # Scale transaction
    # --------------------------------------------------------

    try:

        input_scaled = scaler.transform(input_data)

    except Exception as e:

        st.error("❌ Error while scaling transaction data.")

        st.code(str(e))

        st.stop()


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(input_scaled)[0][1]

    except Exception as e:

        st.error("❌ Error while making prediction.")

        st.code(str(e))

        st.stop()


    # Convert probability to percentage

    fraud_percentage = probability * 100


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if probability >= 0.75:

        risk_level = "HIGH RISK"
        risk_message = "Immediate verification is recommended."

    elif probability >= 0.40:

        risk_level = "MEDIUM RISK"
        risk_message = "Additional verification may be required."

    else:

        risk_level = "LOW RISK"
        risk_message = "Transaction appears relatively safe."


    # ========================================================
    # RESULT HEADER
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">📊 Analysis Result</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # RESULT COLUMNS
    # ========================================================

    result1, result2, result3 = st.columns(3)


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with result1:

        if prediction == 1:

            st.markdown(
                """
                <div class="fraud-result">
                    <div class="result-title">🚨 FRAUD</div>
                    <div class="result-description">
                        Potentially fraudulent transaction
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="safe-result">
                    <div class="result-title">✅ LEGITIMATE</div>
                    <div class="result-description">
                        Transaction appears legitimate
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    with result2:

        st.metric(
            "Fraud Probability",
            f"{fraud_percentage:.2f}%"
        )

        st.progress(
            min(max(float(probability), 0.0), 1.0)
        )


    # --------------------------------------------------------
    # Risk Level
    # --------------------------------------------------------

    with result3:

        st.metric(
            "Risk Level",
            risk_level
        )

        st.write(risk_message)


    # ========================================================
    # DETAILED RESULT
    # ========================================================

    st.markdown(
        '<div class="section-title">📋 Transaction Summary</div>',
        unsafe_allow_html=True
    )


    summary_col1, summary_col2 = st.columns(2)


    with summary_col1:

        st.markdown("#### Transaction Details")

        st.write(f"**Transaction Time:** {time:.2f}")
        st.write(f"**Transaction Amount:** ${amount:,.2f}")
        st.write(f"**Fraud Probability:** {fraud_percentage:.2f}%")


    with summary_col2:

        st.markdown("#### Model Decision")

        if prediction == 1:

            st.error(
                "🚨 The machine learning model has classified "
                "this transaction as potentially fraudulent."
            )

            st.warning(
                "Further verification is recommended before "
                "approving the transaction."
            )

        else:

            st.success(
                "✅ The machine learning model has classified "
                "this transaction as likely legitimate."
            )

            st.info(
                "No strong fraud signal was detected by the model."
            )


    # ========================================================
    # PROBABILITY INTERPRETATION
    # ========================================================

    st.markdown(
        '<div class="section-title">📈 Risk Interpretation</div>',
        unsafe_allow_html=True
    )

    probability_data = pd.DataFrame(
        {
            "Risk Category": [
                "Low Risk",
                "Medium Risk",
                "High Risk"
            ],
            "Probability Range": [
                "0% – 39.99%",
                "40% – 74.99%",
                "75% – 100%"
            ]
        }
    )

    st.dataframe(
        probability_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        💳 Credit Card Fraud Detection System
        <br>
        Machine Learning • Logistic Regression • SMOTE • Streamlit
        <br>
        Developed for educational and demonstration purposes.
    </div>
    """,
    unsafe_allow_html=True
)