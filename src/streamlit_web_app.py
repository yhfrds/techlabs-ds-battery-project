import joblib  # or use import pickle
import numpy as np
import joblib
import streamlit as st
import pandas as pd
import datetime

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
selected_date = st.date_input(
    "### Use Case Date",
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
st.markdown("## Energy and Weather Weather Condition")
st.write(f"📍Freiburg, Baden-Württemberg, Germany")
st.write(f"📅 {formatted_date}")
st.dataframe(df_energy_renamed, hide_index=True)
st.dataframe(df_weather_renamed, hide_index=True)

# Assume df and selected_date are already defined elsewhere in your code

train_button = st.button("Train Model and Show Prediction")

# Check if model has been trained and saved in session state
if train_button or 'model' in st.session_state:

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
        prediction_series = pd.Series(prediction, index=X_test.index)

        # Save the trained model in session state
        st.session_state.model = model
        st.session_state.prediction = prediction
        st.session_state.prediction_series = prediction_series
        st.session_state.y_test = y_test
        st.session_state.train_df = train_df
        st.session_state.test_df = test_df
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.session_state.X_test = X_test

        # Display results
        st.markdown("## Result")
        st.write(f"Training set: {train_df.shape[0]} records")
        st.markdown(
            f"Prediction: <strong>{prediction[0]:.2f} MW</strong>", unsafe_allow_html=True)
        st.markdown(
            f"Actual: <strong>{y_test.values[0]:.2f} MW</strong>", unsafe_allow_html=True)

    st.markdown("## Train-Actual-Prediction Values Comparison")
    min_date = df.index.min().date()
    max_date = df.index.max().date()

    start_date = st.date_input(
        "Select start date for chart",
        min_value=min_date,
        max_value=max_date,
        value=max(selected_date - datetime.timedelta(weeks=2), min_date),
        format="YYYY-MM-DD"
    )
    end_date = st.date_input(
        "Select end date for chart",
        min_value=min_date,
        max_value=max_date,
        value=min(selected_date + datetime.timedelta(weeks=1), max_date),
        format="YYYY-MM-DD"
    )

    model = st.session_state.model
    prediction = st.session_state.prediction
    prediction_series = st.session_state.prediction_series
    y_test = st.session_state.y_test
    train_df = st.session_state.train_df
    test_df = st.session_state.test_df
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test

    # Create a DataFrame for plotting
    chart_data = pd.concat([
        y_train.rename("actual_train"),
        y_test.rename("actual_test"),
        prediction_series.rename("prediction")
    ], axis=1).replace(0, np.nan)

    # Filter data based on selected date range
    chart_data = chart_data[(chart_data.index >= pd.to_datetime(start_date)) &
                            (chart_data.index <= pd.to_datetime(end_date))]
    st.line_chart(chart_data)
