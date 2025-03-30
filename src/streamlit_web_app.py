r"""
Before running this code, ensure that Streamlit is installed. You can install it using the following command:

    pip install streamlit

Next, it's recommended to add Streamlit to your system PATH. You can do this by running:

    $env:Path += ";C:\Users\yudha\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\..."

Finally, to launch the Streamlit app, use:

    streamlit run streamlit_web_app.py

If the command doesn't work, double-check that you are in the correct directory in your command prompt.
"""


import numpy as np
import joblib
import streamlit as st
import pandas as pd

st.write("""
# ⚡ Smart Grid Balancer: Residual Load Forecasting    

🔍 Explore different forecasting models and see how data-driven predictions can optimize renewable energy planning! 
""")

st.write('---')

st.header('TransnetBW')
st.subheader("Cities in the TransnetBW Region")
cities = [
    "Stuttgart", "Karlsruhe", "Mannheim", "Freiburg", "Heidelberg",
    "Ulm", "etc."
]
st.write(", ".join(cities))
st.image(image='../data/images/TransnetBW area.png',
         caption="TransnetBW Coverage", use_container_width=True)

st.write('---')

st.header('Forecasting App')


# Date input for start and end date selection
st.subheader("Select Use Case Date")
selected_date = st.date_input(
    "Use Case Date", min_value='2018-06-18', max_value='2025-02-02', value='2025-01-15')
# Convert selected_date to match DataFrame index format
selected_date_str = selected_date.strftime('%Y-%m-%d')
formatted_date = selected_date.strftime("%A, %d %B, %Y")


df = pd.read_csv("../data/final_dataset.csv")
df.set_index("Date", inplace=True)

selected_energy_features = ['Total_Load', 'Electricity_Generated_Wind',
                            'Electricity_Generated_Photovoltaics', 'Residual_Load']
selected_weather_features = ['Air_Temperature', 'Relative_Humidity',
                             'Air_Pressure_at_Station_Height', 'Cloud_Cover',
                             'Daily_Precipitation_Height',
                             'Global_Radiation']

# Define new feature names
energy_feature_names = {
    'Total_Load': 'Total Energy Consumption (MW)',
    'Electricity_Generated_Wind': 'Wind Power Generation (MW)',
    'Electricity_Generated_Photovoltaics': 'Solar Power Generation (MW)',
    'Residual_Load': 'Required Non-Renewable Generation (MW)'
}

weather_feature_names = {
    'Air_Temperature': 'Temperature (°C)',
    'Relative_Humidity': 'Humidity (%)',
    'Air_Pressure_at_Station_Height': 'Pressure (hPa)',
    'Cloud_Cover': 'Cloud Coverage (%)',
    'Daily_Precipitation_Height': 'Precipitation (mm)',
    'Global_Radiation': 'Solar Radiation (W/m²)'
}

# Rename columns
df_energy_renamed = df[selected_energy_features].loc[[selected_date_str]].rename(
    columns=energy_feature_names)
df_weather_renamed = df[selected_weather_features].loc[[selected_date_str]].rename(
    columns=weather_feature_names)

# Streamlit UI
st.subheader("Energy Usage and Generation")
st.write(f"📅 {formatted_date}")
st.dataframe(df_energy_renamed)

st.subheader("Weather Condition")
st.write(f"📅 {formatted_date}")
st.dataframe(df_weather_renamed)
