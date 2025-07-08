import streamlit as st
from purchase_model import load_and_prepare_data, train_and_evaluate
import pandas as pd

# 🧼 Page setup
st.set_page_config(page_title="🛍️ Purchase Intention Predictor", layout="centered")

# 🎨 CSS styling (optional)
st.markdown("""
<style>
    .big-font { font-size: 30px !important; font-weight: 600; }
    .subheader { font-size: 20px; font-weight: 500; color: #4ecdc4; margin-top: 20px; }
    .reportbox { background-color: #222; padding: 20px; border-radius: 10px; }
    .stButton > button {
        background-color: #4ecdc4;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# 🔰 Title
st.markdown("<div class='big-font'>🛍️ Purchase Intention Predictor</div>", unsafe_allow_html=True)
st.markdown("Use machine learning to predict whether a user will make a purchase based on their session behavior.")

# 📤 Upload
st.markdown("### 📥 Upload Your Dataset (.csv)")
uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file:
    df = load_and_prepare_data(uploaded_file)

    with st.expander("📊 View Dataset (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)

    # 🔍 Train Model
    if st.button("🚀 Train Model"):
        model, metrics, features = train_and_evaluate(df)
        st.session_state['model'] = model
        st.session_state['features'] = features

        st.markdown("### 📈 Model Performance")
        st.markdown("<div class='reportbox'>", unsafe_allow_html=True)
        st.write(f"**ROC AUC:** `{metrics['roc_auc']:.4f}`")
        st.write(f"**Accuracy:** `{metrics['accuracy']:.4f}`")
        st.write(f"**F1 Score:** `{metrics['f1_score']:.4f}`")
        st.text("Classification Report:")
        st.text(metrics['report'])
        st.markdown("</div>", unsafe_allow_html=True)

    # 🔮 Prediction Section
    if 'model' in st.session_state:
        st.markdown("### 🔮 Predict Purchase Intention")
        st.markdown("Provide basic session details to predict if user will purchase.")

        with st.form("predict_form"):
            col1, col2 = st.columns(2)

            with col1:
                duration = st.slider("🕒 Time on Product Pages (secs)", 0, 1000, 300)
                bounce_rate = st.slider("📉 Bounce Rate", 0.0, 1.0, 0.02)

            with col2:
                page_value = st.number_input("🌐 Page Value", value=20.0)
                weekend = st.radio("📅 Weekend Visit?", [0, 1], format_func=lambda x: "Yes" if x else "No")
                returning = st.radio("🔁 Returning Visitor?", [0, 1], format_func=lambda x: "Yes" if x else "No")

            submit = st.form_submit_button("🔎 Predict")

        if submit:
            sample = {
                'Administrative': 1,
                'Administrative_Duration': 20,
                'Informational': 1,
                'Informational_Duration': 10,
                'ProductRelated': 10,
                'ProductRelated_Duration': duration,
                'BounceRates': bounce_rate,
                'ExitRates': 0.04,
                'PageValues': page_value,
                'SpecialDay': 0.0,
                'Month': 6,
                'OperatingSystems': 2,
                'Browser': 2,
                'Region': 1,
                'TrafficType': 1,
                'Weekend': weekend,
                'Returning_Visitor': returning
            }

            input_df = pd.DataFrame([sample])
            prediction = st.session_state['model'].predict(input_df)[0]

            st.success(f"🎯 Prediction: {'✅ Will Purchase' if prediction == 1 else '❌ Will Not Purchase'}")
