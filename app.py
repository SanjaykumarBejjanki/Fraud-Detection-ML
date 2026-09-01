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
# CONSTANTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
METRICS_PATH = BASE_DIR / "outputs" / "metrics.txt"

FEATURES = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]


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

    /* Main content width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Header */
    .hero {
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #111827, #1f2937);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    }

    .hero h1 {
        font-size: 2.5rem;
        margin-bottom: 0.4rem;
    }

    .hero p {
        color: #d1d5db;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    /* Cards */
    .metric-card {
        background: white;
        padding: 1.3rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(0,0,0,0.05);
        min-height: 120px;
    }

    .metric-title {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        color: #111827;
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Section titles */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #111827;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* Risk cards */
    .risk-high {
        padding: 1.5rem;
        border-radius: 16px;
        background: #fee2e2;
        border: 1px solid #fecaca;
    }

    .risk-medium {
        padding: 1.5rem;
        border-radius: 16px;
        background: #fef3c7;
        border: 1px solid #fde68a;
    }

    .risk-low {
        padding: 1.5rem;
        border-radius: 16px;
        background: #dcfce7;
        border: 1px solid #bbf7d0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 45px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        padding-top: 2rem;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


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
    st.error("Unable to load the machine learning model.")
    st.code(str(e))
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:10px;">
            <div style="font-size:3rem;">💳</div>
            <h2>Fraud Detection AI</h2>
            <p style="color:#9ca3af;">
                Intelligent transaction risk analysis
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔍 Transaction Analyzer",
            "📁 Batch Prediction",
            "📊 Model Performance"
        ]
    )

    st.divider()

    st.markdown("### 🤖 Model")
    st.write("Algorithm: Logistic Regression")
    st.write("Imbalance handling: SMOTE")
    st.write("Features: 30")

    st.divider()

    st.caption("Fraud Detection AI")
    st.caption("Machine Learning + Streamlit")


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>💳 Fraud Detection AI</h1>
        <p>
            AI-powered credit card transaction risk analysis
            using machine learning and anomaly detection.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">📈 Model Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">ROC-AUC</div>
                <div class="metric-value">97.10%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Recall</div>
                <div class="metric-value">91.84%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Precision</div>
                <div class="metric-value">5.79%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">F1 Score</div>
                <div class="metric-value">10.89%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # Dataset statistics
    st.markdown(
        '<div class="section-title">💳 Dataset Statistics</div>',
        unsafe_allow_html=True
    )

    stat1, stat2, stat3, stat4 = st.columns(4)

    with stat1:
        st.metric("Total Transactions", "284,807")

    with stat2:
        st.metric("Legitimate", "284,315")

    with stat3:
        st.metric("Fraudulent", "492")

    with stat4:
        st.metric("Fraud Rate", "0.173%")

    st.write("")

    # Class distribution
    chart_data = pd.DataFrame(
        {
            "Transaction Type": [
                "Legitimate",
                "Fraudulent"
            ],
            "Count": [
                284315,
                492
            ]
        }
    )

    st.markdown(
        '<div class="section-title">📊 Transaction Distribution</div>',
        unsafe_allow_html=True
    )

    st.bar_chart(
        chart_data.set_index("Transaction Type")
    )

    st.info(
        "⚠️ Fraud detection is an imbalanced classification problem. "
        "Recall is especially important because missing a fraudulent "
        "transaction can be costly."
    )


# ============================================================
# TRANSACTION ANALYZER
# ============================================================

elif page == "🔍 Transaction Analyzer":

    st.markdown(
        '<div class="section-title">🔍 Analyze a Transaction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the transaction's numerical features below. "
        "The trained model will estimate the probability of fraud."
    )

    with st.form("transaction_form"):

        st.markdown("### Transaction Information")

        col1, col2 = st.columns(2)

        with col1:
            time = st.number_input(
                "Transaction Time",
                min_value=0.0,
                value=0.0,
                help="Time elapsed since the first transaction in the dataset."
            )

        with col2:
            amount = st.number_input(
                "💰 Transaction Amount",
                min_value=0.0,
                value=100.0,
                step=1.0
            )

        st.markdown("### 🔐 PCA Features")

        col1, col2, col3 = st.columns(3)

        with col1:
            v1 = st.number_input("V1", value=0.0)
            v2 = st.number_input("V2", value=0.0)
            v3 = st.number_input("V3", value=0.0)
            v4 = st.number_input("V4", value=0.0)
            v5 = st.number_input("V5", value=0.0)
            v6 = st.number_input("V6", value=0.0)
            v7 = st.number_input("V7", value=0.0)
            v8 = st.number_input("V8", value=0.0)
            v9 = st.number_input("V9", value=0.0)

        with col2:
            v10 = st.number_input("V10", value=0.0)
            v11 = st.number_input("V11", value=0.0)
            v12 = st.number_input("V12", value=0.0)
            v13 = st.number_input("V13", value=0.0)
            v14 = st.number_input("V14", value=0.0)
            v15 = st.number_input("V15", value=0.0)
            v16 = st.number_input("V16", value=0.0)
            v17 = st.number_input("V17", value=0.0)
            v18 = st.number_input("V18", value=0.0)

        with col3:
            v19 = st.number_input("V19", value=0.0)
            v20 = st.number_input("V20", value=0.0)
            v21 = st.number_input("V21", value=0.0)
            v22 = st.number_input("V22", value=0.0)
            v23 = st.number_input("V23", value=0.0)
            v24 = st.number_input("V24", value=0.0)
            v25 = st.number_input("V25", value=0.0)
            v26 = st.number_input("V26", value=0.0)
            v27 = st.number_input("V27", value=0.0)
            v28 = st.number_input("V28", value=0.0)

        analyze = st.form_submit_button(
            "🚨 Analyze Transaction",
            use_container_width=True
        )

    if analyze:

        input_data = pd.DataFrame(
            [[
                time,
                v1, v2, v3, v4, v5, v6, v7, v8, v9,
                v10, v11, v12, v13, v14, v15, v16, v17, v18,
                v19, v20, v21, v22, v23, v24, v25, v26, v27, v28,
                amount
            ]],
            columns=FEATURES
        )

        try:

            # IMPORTANT:
            # DataFrame with feature names removes the
            # StandardScaler feature-name warning.
            input_scaled = scaler.transform(input_data)

            prediction = int(model.predict(input_scaled)[0])

            probability = float(
                model.predict_proba(input_scaled)[0][1]
            )

            st.divider()

            st.markdown(
                '<div class="section-title">📊 Analysis Result</div>',
                unsafe_allow_html=True
            )

            result1, result2 = st.columns(2)

            with result1:

                if prediction == 1:
                    st.error(
                        "🚨 FRAUDULENT TRANSACTION"
                    )
                else:
                    st.success(
                        "✅ LEGITIMATE TRANSACTION"
                    )

            with result2:

                st.metric(
                    "Fraud Probability",
                    f"{probability * 100:.2f}%"
                )

            st.progress(probability)

            # Risk level
            if probability >= 0.75:

                st.markdown(
                    """
                    <div class="risk-high">
                        <h3>🔴 HIGH RISK</h3>
                        <p>
                            This transaction has a high estimated
                            probability of fraud. Additional verification
                            is strongly recommended.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif probability >= 0.40:

                st.markdown(
                    """
                    <div class="risk-medium">
                        <h3>🟡 MEDIUM RISK</h3>
                        <p>
                            This transaction shows suspicious characteristics.
                            Consider additional verification.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="risk-low">
                        <h3>🟢 LOW RISK</h3>
                        <p>
                            The model considers this transaction likely
                            legitimate.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        except Exception as e:

            st.error("Prediction failed.")
            st.code(str(e))


# ============================================================
# BATCH PREDICTION
# ============================================================

elif page == "📁 Batch Prediction":

    st.markdown(
        '<div class="section-title">📁 Batch Fraud Detection</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a CSV file containing the same 30 features used "
        "during model training."
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.success(
                f"File uploaded successfully: {len(df):,} transactions"
            )

            st.markdown("### 👀 Data Preview")

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            missing_features = [
                feature for feature in FEATURES
                if feature not in df.columns
            ]

            if missing_features:

                st.error(
                    "The uploaded CSV is missing required features:"
                )

                st.code(
                    ", ".join(missing_features)
                )

            else:

                prediction_input = df[FEATURES].copy()

                # Convert all features to numeric
                prediction_input = prediction_input.apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                if prediction_input.isnull().any().any():

                    st.error(
                        "The uploaded file contains missing or "
                        "non-numeric values in the required features."
                    )

                else:

                    if st.button(
                        "🚨 Analyze All Transactions",
                        use_container_width=True
                    ):

                        scaled_data = scaler.transform(
                            prediction_input
                        )

                        predictions = model.predict(
                            scaled_data
                        )

                        probabilities = model.predict_proba(
                            scaled_data
                        )[:, 1]

                        result_df = df.copy()

                        result_df["Fraud Prediction"] = predictions

                        result_df["Fraud Probability"] = (
                            probabilities * 100
                        ).round(2)

                        result_df["Risk Level"] = np.select(
                            [
                                probabilities >= 0.75,
                                probabilities >= 0.40
                            ],
                            [
                                "High",
                                "Medium"
                            ],
                            default="Low"
                        )

                        fraud_count = int(
                            (predictions == 1).sum()
                        )

                        legitimate_count = int(
                            (predictions == 0).sum()
                        )

                        fraud_percentage = (
                            fraud_count / len(predictions) * 100
                            if len(predictions) > 0
                            else 0
                        )

                        st.divider()

                        a, b, c = st.columns(3)

                        with a:
                            st.metric(
                                "Total Transactions",
                                f"{len(result_df):,}"
                            )

                        with b:
                            st.metric(
                                "Fraudulent",
                                f"{fraud_count:,}"
                            )

                        with c:
                            st.metric(
                                "Fraud Rate",
                                f"{fraud_percentage:.2f}%"
                            )

                        st.markdown(
                            "### 📊 Prediction Results"
                        )

                        st.dataframe(
                            result_df,
                            use_container_width=True
                        )

                        csv = result_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="📥 Download Prediction Results",
                            data=csv,
                            file_name="fraud_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

        except Exception as e:

            st.error(
                "Unable to process the uploaded CSV."
            )

            st.code(str(e))


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.markdown(
        '<div class="section-title">📊 Model Performance</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Performance metrics from the trained Logistic Regression "
        "fraud detection model."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "ROC-AUC",
            "97.10%"
        )

    with col2:
        st.metric(
            "Precision",
            "5.79%"
        )

    with col3:
        st.metric(
            "Recall",
            "91.84%"
        )

    with col4:
        st.metric(
            "F1 Score",
            "10.89%"
        )

    st.divider()

    st.markdown("### 🎯 Confusion Matrix")

    confusion_matrix = pd.DataFrame(
        {
            "Predicted Legitimate": [
                55399,
                8
            ],
            "Predicted Fraud": [
                1465,
                90
            ]
        },
        index=[
            "Actual Legitimate",
            "Actual Fraud"
        ]
    )

    st.dataframe(
        confusion_matrix,
        use_container_width=True
    )

    st.markdown("### 📋 Classification Summary")

    report = pd.DataFrame(
        {
            "Class": [
                "Legitimate",
                "Fraudulent"
            ],
            "Precision": [
                1.00,
                0.0579
            ],
            "Recall": [
                0.97,
                0.9184
            ],
            "F1 Score": [
                0.99,
                0.1089
            ]
        }
    )

    st.dataframe(
        report.style.format(
            {
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 Score": "{:.2%}"
            }
        ),
        use_container_width=True
    )

    st.info(
        "The model prioritizes fraud detection recall. "
        "This means it successfully identifies most fraudulent "
        "transactions, but the relatively low precision means "
        "some legitimate transactions are also flagged."
    )

    if METRICS_PATH.exists():

        with st.expander("📄 View training metrics"):

            try:
                metrics_text = METRICS_PATH.read_text(
                    encoding="utf-8"
                )

                st.code(metrics_text)

            except Exception:
                st.warning(
                    "Metrics file could not be read."
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        💳 <strong>Fraud Detection AI</strong><br>
        Machine Learning • Python • Scikit-learn • SMOTE • Streamlit<br>
        Built for intelligent transaction risk analysis
    </div>
    """,
    unsafe_allow_html=True
)