import streamlit as st
import pandas as pd
from model import train_model

st.set_page_config(page_title="Traffic Volume Predictor", layout="centered")

st.title("🚦 Metro Traffic Volume Predictor")

st.markdown("This ML app predicts **traffic volume** based on weather and time features.")

with st.spinner("Training model..."):
    model, metrics = train_model()

st.success("Model trained successfully!")

st.subheader("📊 Model Accuracy Metrics")
st.table(pd.DataFrame([metrics]))

st.markdown("---")

st.subheader("🔍 Predict Traffic Volume")

temp = st.slider("Temperature (in Kelvin)", 250, 320, 280)
rain_1h = st.slider("Rain (mm)", 0.0, 50.0, 0.0)
snow_1h = st.slider("Snow (mm)", 0.0, 50.0, 0.0)
clouds_all = st.slider("Clouds (%)", 0, 100, 50)
hour = st.slider("Hour of Day", 0, 23, 12)
month = st.slider("Month", 1, 12, 6)
dayofweek = st.slider("Day of Week (0=Mon)", 0, 6, 2)

weather_main = st.selectbox("Weather Main", ['Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze', 'Mist', 'Rain', 'Smoke', 'Snow', 'Squall', 'Thunderstorm'])

# One-hot encoding of weather
weather_columns = ['weather_main_Clear', 'weather_main_Clouds', 'weather_main_Drizzle',
                   'weather_main_Fog', 'weather_main_Haze', 'weather_main_Mist',
                   'weather_main_Rain', 'weather_main_Smoke', 'weather_main_Snow',
                   'weather_main_Squall', 'weather_main_Thunderstorm']

weather_data = {col: 0 for col in weather_columns}
col_name = f"weather_main_{weather_main}"
if col_name in weather_data:
    weather_data[col_name] = 1

# Final input
input_data = pd.DataFrame([{
    'temp': temp,
    'rain_1h': rain_1h,
    'snow_1h': snow_1h,
    'clouds_all': clouds_all,
    'hour': hour,
    'month': month,
    'dayofweek': dayofweek,
    **weather_data
}])

if st.button("Predict Traffic Volume"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Traffic Volume: {int(prediction)} vehicles")
