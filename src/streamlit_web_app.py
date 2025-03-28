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

st.write("""
# ⚡ Smart Grid Balancer: Residual Load Forecasting    

🔍 Explore different forecasting models and see how data-driven predictions can optimize renewable energy planning! 


""")

st.write("helooooo")
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


# Side Bar
st.sidebar.header("Possible Parameters")

st.sidebar.subheader("Select Forecasting Duration")

col1, col2 = st.sidebar.columns([1, 2])

with col1:
    # Default time unit to "Days" before defining number input
    future_value = st.number_input(
        "Future:", min_value=1, max_value=365, value=7)

with col2:
    time_unit = st.selectbox("Unit:", ["Days", "Weeks", "Months"])

# Adjust max values dynamically based on selected time unit
if time_unit == "Weeks":
    future_value = st.sidebar.number_input(
        "Future:", min_value=1, max_value=52, value=1)
elif time_unit == "Months":
    future_value = st.sidebar.number_input(
        "Future:", min_value=1, max_value=12, value=1)


# Sidebar for date selection
st.sidebar.header("Choose Time Period for Result Chart")

# Date input for start and end date selection
start_date = st.sidebar.date_input("Start date")
end_date = st.sidebar.date_input(
    "End date")

# Display the selected date range
st.write(f"Showing data from {start_date} to {end_date}")


# Load model
model = joblib.load("final_xgb_model.pkl")

# Example usage (assuming it's a regression model)
X_test = np.array([[5, 3, 1.5, 0.2]])  # Example input
prediction = model.predict(X_test)

print("Prediction:", prediction)
