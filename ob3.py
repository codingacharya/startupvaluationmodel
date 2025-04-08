import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv("startup_data.csv")  # Replace with your real CSV

# Feature configuration
categorical_cols = ['industry', 'exit_status']
numerical_cols = ['funding_rounds', 'valuation_usd', 'user_growth_rate', 'revenue_growth_rate',
                  'macro_trend_index', 'tech_edge_score', 'partnerships_score']

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Handle missing values
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Features and targets
X = df[numerical_cols + ['industry']]
y_valuation = df['valuation_usd']
y_exit = df['exit_status']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, _, y_val_train, _ = train_test_split(X_scaled, y_valuation, test_size=0.2, random_state=42)
X_train_cls, _, y_exit_train, _ = train_test_split(X_scaled, y_exit, test_size=0.2, random_state=42)

# Train models
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train, y_val_train)

cls_model = RandomForestClassifier(n_estimators=100, random_state=42)
cls_model.fit(X_train_cls, y_exit_train)

# Streamlit UI
st.set_page_config(page_title="Startup Valuation & Exit Predictor", layout="centered")
st.title("🔮 Startup Valuation & Exit Strategy Predictor")
st.markdown("Predict future **valuation** and **exit potential** (IPO, Acquisition, Failure) of startups.")

# User input
with st.form("input_form"):
    funding_rounds = st.number_input("Funding Rounds", min_value=0, step=1, value=2)
    valuation = st.number_input("Current Valuation (USD)", min_value=0, value=5000000)
    user_growth = st.slider("User Growth Rate", 0.0, 1.0, 0.15)
    revenue_growth = st.slider("Revenue Growth Rate", 0.0, 1.0, 0.20)
    macro_index = st.slider("Macro Trend Index", 0.0, 1.0, 0.75)
    tech_edge = st.slider("Tech Edge Score", 0.0, 1.0, 0.80)
    partnership_score = st.slider("Partnership Score", 0.0, 1.0, 0.70)
    industry = st.selectbox("Industry", df['industry'].unique())
    submitted = st.form_submit_button("Predict")

# Prediction
if submitted:
    input_data = {
        'funding_rounds': funding_rounds,
        'valuation_usd': valuation,
        'user_growth_rate': user_growth,
        'revenue_growth_rate': revenue_growth,
        'macro_trend_index': macro_index,
        'tech_edge_score': tech_edge,
        'partnerships_score': partnership_score,
        'industry': industry
    }

    input_df = pd.DataFrame([input_data])
    input_df['industry'] = label_encoders['industry'].transform(input_df['industry'])
    input_df[numerical_cols] = input_df[numerical_cols].fillna(df[numerical_cols].mean())
    input_scaled = scaler.transform(input_df[numerical_cols + ['industry']])

    predicted_valuation = reg_model.predict(input_scaled)[0]
    predicted_exit = cls_model.predict(input_scaled)[0]
    exit_label = label_encoders['exit_status'].inverse_transform([predicted_exit])[0]

    st.subheader("📈 Predicted Results")
    st.success(f"💰 Estimated Future Valuation: **${predicted_valuation:,.2f}**")
    st.info(f"🚀 Likely Exit Strategy: **{exit_label}**")
