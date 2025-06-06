import streamlit as st
import pandas as pd
import pickle

# Page config
st.set_page_config(page_title="Traffic Volume Predictor", layout="wide")

# Load the model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Load the dataset (used in model.py)
@st.cache_data
def load_data():
    df = pd.read_csv("traffic_data_800.csv")
    df.drop(columns=["date_time", "weather_description"], inplace=True, errors="ignore")
    df = pd.get_dummies(df, columns=["holiday", "weather_main"], drop_first=True)
    df.dropna(inplace=True)
    return df

# Load feature sample for form
def get_feature_sample():
    return load_data().drop(columns=["traffic_volume"], errors="ignore")

# Sidebar navigation
st.sidebar.title("Go to")
page = st.sidebar.radio("", ["Home", "Dataset", "Summary", "Predict"])

# =================== Home ===================
if page == "Home":
    st.title("🚦 Traffic Volume Prediction App")
    st.markdown("""
        Welcome to the **Traffic Volume Predictor**!
        
        This app uses a trained **Random Forest Regression** model to estimate traffic volume based on weather and holiday data.
        
        Use the sidebar to navigate through:
        - 📊 Dataset view
        - 📈 Summary statistics
        - 🔍 Live prediction form
    """)

# =================== Dataset ===================
elif page == "Dataset":
    st.title("📊 Dataset Preview")
    df = load_data()
    st.dataframe(df.head(50), use_container_width=True)
    st.caption(f"Showing first 50 of {len(df)} rows from `traffic_data_800.csv`")

# =================== Summary ===================
elif page == "Summary":
    st.title("📈 Dataset Summary")
    df = load_data()
    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", df.columns.tolist())
    st.bar_chart(df["traffic_volume"])

# =================== Predict ===================
elif page == "Predict":
    st.title("🎯 Predict Traffic Volume")

    st.markdown("Enter input values below to get a prediction:")

    df_sample = get_feature_sample()
    input_data = {}

    # 2-column layout for input fields
    col1, col2 = st.columns(2)
    for i, col in enumerate(df_sample.columns):
        with (col1 if i % 2 == 0 else col2):
            if pd.api.types.is_numeric_dtype(df_sample[col]):
                input_data[col] = st.number_input(col, value=float(df_sample[col].median()))
            else:
                input_data[col] = st.selectbox(col, list(df_sample[col].unique()))

    st.markdown("---")
    if st.button("🔍 Predict"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.success(f"🚗 **Predicted Traffic Volume:** {int(prediction):,} vehicles")
