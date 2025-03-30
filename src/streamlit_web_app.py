import joblib  # or use import pickle
import numpy as np
import joblib
import streamlit as st
import pandas as pd

st.write("""
# ⚡ Smart Grid Balancer: Residual Load Forecasting    

🔍 Explore different forecasting models and see how data-driven predictions can optimize renewable energy planning! 
""")

st.write('---')

st.markdown('# TransnetBW')
st.markdown("#### Cities in the TransnetBW Region")
cities = [
    "Stuttgart", "Karlsruhe", "Mannheim", "Freiburg", "Heidelberg",
    "Ulm", "etc."
]
st.write(", ".join(cities))
st.image(image='../data/images/TransnetBW area.png',
         caption="TransnetBW Coverage", use_container_width=True)

st.write('---')

st.markdown('# XGBoost Timeseries Forecasting')


# Date input for start and end date selection
st.markdown("##### Select Use Case Date")
selected_date = st.date_input(
    "Date",
    min_value=pd.to_datetime('2018-06-18').date(),
    max_value=pd.to_datetime('2025-02-02').date(),
    value=pd.to_datetime('2025-01-15').date()
)
# Convert selected_date to match DataFrame index format
selected_date_str = selected_date.strftime('%Y-%m-%d')
formatted_date = selected_date.strftime("%A, %d %B, %Y")

df = pd.read_csv("../data/final_dataset.csv")
df.set_index("Date", inplace=True)
df.index = pd.to_datetime(df.index)

selected_energy_features = ['Total_Load', 'Electricity_Generated_Wind',
                            'Electricity_Generated_Photovoltaics', 'Residual_Load']
selected_weather_features = ['Air_Temperature', 'Relative_Humidity',
                             'Air_Pressure_at_Station_Height', 'Cloud_Cover',
                             'Daily_Precipitation_Height',
                             'Global_Radiation', 'wind_u', 'wind_v']

# Define new feature names
energy_feature_names = {
    'Total_Load': 'Total Grid Load (MW)',
    'Electricity_Generated_Wind': 'Wind Power Generation (MW)',
    'Electricity_Generated_Photovoltaics': 'Solar Power Generation (MW)',
    'Residual_Load': 'Residual Load (MW)'
}

weather_feature_names = {
    'Air_Temperature': '🌡️ Temperature (°C)',
    'Relative_Humidity': '💧 Humidity (%)',
    'Air_Pressure_at_Station_Height': '💨 Pressure (hPa)',
    'Cloud_Cover': '☁️Cloud Coverage (%)',
    'Daily_Precipitation_Height': '🌧️Precipitation (mm)',
    'Global_Radiation': '☀️ Solar Radiation (W/m²)'
}

# Rename columns
df_energy_renamed = df[selected_energy_features].loc[[selected_date_str]].rename(
    columns=energy_feature_names)
df_weather_renamed = df[selected_weather_features].loc[[selected_date_str]].rename(
    columns=weather_feature_names)
df_weather_renamed['🍃 Wind (m/s)'] = np.sqrt(df_weather_renamed['wind_u']
                                             ** 2 + df_weather_renamed['wind_v']**2)
selected_weather_features_final = ['🌡️ Temperature (°C)', '💧 Humidity (%)', '☀️ Solar Radiation (W/m²)',
                                   '🍃 Wind (m/s)', '☁️Cloud Coverage (%)', '🌧️Precipitation (mm)', '💨 Pressure (hPa)']
df_weather_renamed = df_weather_renamed[selected_weather_features_final]

# Streamlit UI
st.markdown("### Energy and Weather Weather Condition")
st.write(f"📍Freiburg, Baden-Württemberg, Germany")
st.write(f"📅 {formatted_date}")
st.dataframe(df_energy_renamed, hide_index=True)
st.dataframe(df_weather_renamed, hide_index=True)
train_button = st.button("Train Model and Show Prediction")


if train_button:
    with st.spinner('Training model...'):
        # Train and Test Set
        train_df = df[df.index < selected_date_str]
        test_df = df[df.index >= selected_date_str]

        X_train = train_df.drop(columns=['Residual_Load_Tomorrow'])
        y_train = train_df['Residual_Load_Tomorrow']
        X_test = test_df.drop(columns=['Residual_Load_Tomorrow'])
        y_test = test_df['Residual_Load_Tomorrow']

        # Load the pre-trained model
        model = joblib.load('../src/final_xgb_model.pkl')
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        # Displaying the results in a more engaging way
        st.markdown("### Result")
        st.write(f"Training set: {train_df.shape[0]} records")
        st.markdown(
            f"Prediction: <strong>{prediction[0]:.2f} MW</strong>", unsafe_allow_html=True)
        st.markdown(
            f"Actual: <strong>{y_test.values[0]:.2f} MW</strong>", unsafe_allow_html=True)
        st.success("Training completed!")
