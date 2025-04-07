import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# App Configuration
# App Header
st.write("""
# ⚡ Smart Grid Balancer: Residual Load Forecasting (Auto-arima Timeseries Forecasting)
🔍 Explore different forecasting models and see how data-driven predictions can optimize renewable energy planning!
""")

st.write('---')

# TransnetBW Section
st.header('TransnetBW')
st.subheader("Cities in the TransnetBW Region")
cities = ["Stuttgart", "Karlsruhe", "Mannheim", "Freiburg", "Heidelberg", "Ulm"]
st.write(", ".join(cities))
try:
    image_path = r"C:\Users\sinyi\Documents\1_Techlabs Residual Load Project\TransnetBW_area.png"
    st.image(image=image_path,
             caption="TransnetBW Coverage",
             use_container_width=True)
except FileNotFoundError:
    st.warning("Coverage map image not found")

st.write('---')

# Data Loading
@st.cache_data
def load_data():
    df = pd.read_csv('freiburg_residual_load.csv', sep=';', 
                    index_col='Date', parse_dates=True)
    df.rename(columns={df.columns[1]: 'residual_load'}, inplace=True)
    return df[['residual_load']]

df = load_data()

# Date Selection
col1, col2 = st.columns(2)
with col1:
    selected_date = st.date_input(
        "Forecast Start Date",
        min_value=df.index.min().date(),
        max_value=df.index.max().date(),
        value=df.index.max().date() - pd.Timedelta(days=7)
    )
with col2:
    forecast_days = st.slider("Forecast Horizon (days)", 1, 14, 7)

selected_date_str = str(selected_date)

# Model Training
if st.button("Generate Forecast"):
    with st.spinner('Training SARIMAX model...'):
        train_data = df.loc[:selected_date_str]
        test_dates = pd.date_range(start=selected_date_str, periods=forecast_days)
        
        # Check if test data exists
        test_data = df.loc[selected_date_str:test_dates[-1]] if selected_date_str in df.index else None
        
        # Train model
        model = SARIMAX(
            train_data,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 2, 7)
        )
        results = model.fit()
        predictions = results.get_forecast(steps=forecast_days).predicted_mean
        
        # Store results
        st.session_state['forecast'] = {
            'history': train_data,
            'predictions': predictions,
            'actuals': test_data,
            'dates': test_dates
        }
        st.success("Forecast generated!")
  
# Visualization
if 'forecast' in st.session_state:
    st.markdown("## Forecast Results")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot historical data
    ax.plot(st.session_state['forecast']['history'].index[-90:], 
            st.session_state['forecast']['history']['residual_load'][-90:],
            label='Historical Data', color='#1f77b4', linewidth=2)
    
    # Plot forecast
    ax.plot(st.session_state['forecast']['dates'],
            st.session_state['forecast']['predictions'],
            label='Forecast', color='#ff7f0e', linewidth=2, linestyle='--')
    
    # Plot actuals if available
    if st.session_state['forecast']['actuals'] is not None:
        ax.plot(st.session_state['forecast']['actuals'].index,
                st.session_state['forecast']['actuals']['residual_load'],
                label='Actual Values', color='#2ca02c', linewidth=2)
        
        # Calculate metrics
        actuals = st.session_state['forecast']['actuals']['residual_load'].values
        preds = st.session_state['forecast']['predictions'].values[:len(actuals)]
        
        metrics = {
            'MAE': mean_absolute_error(actuals, preds) / 1000,  # Convert to GW
            'RMSE': np.sqrt(mean_squared_error(actuals, preds)) / 1000,  # Convert to GW
            'MAPE': mean_absolute_percentage_error(actuals, preds)*100
        }
        
        # Display metrics
        st.markdown("### Forecast Accuracy")
        cols = st.columns(3)
        cols[0].metric("Mean Absolute Error", f"{metrics['MAE']:.2f} GW")
        cols[1].metric("Root Mean Squared Error", f"{metrics['RMSE']:.2f} GW")
        cols[2].metric("Mean Absolute % Error", f"{int(round(metrics['MAPE']))}%")
    
    # Formatting
    ax.set_title('Residual Load Forecast', pad=20, fontsize=14)
    ax.set_xlabel('Date', labelpad=10)
    ax.set_ylabel('Load (MW)', labelpad=10)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Data Table
    st.markdown("### Forecast Data")
    forecast_df = pd.DataFrame({
        'Date': st.session_state['forecast']['dates'],
        'Predicted (MW)': st.session_state['forecast']['predictions'].values
    }).set_index('Date')
    
    if st.session_state['forecast']['actuals'] is not None:
        forecast_df['Actual (MW)'] = st.session_state['forecast']['actuals']['residual_load'].values
    
    st.dataframe(forecast_df.style.format("{:.2f}"), use_container_width=True)
    
    # Evaluation Metrics (now after the data table)
    st.markdown("### Evaluation Metrics of Model Throughout Entire Dataset")
    cols = st.columns(3)
    with cols[0]:
        st.metric("Mean Absolute Error", "14.68 GW",  # Changed to 2 SF and GW
                 help="Average absolute error between forecast and actual values")
    with cols[1]:
        st.metric("Root Mean Squared Error", "18.68 GW",  # Changed to 2 SF and GW
                 help="Square root of the average of squared errors")
    with cols[2]:
        st.metric("Mean Absolute % Error", "12%",
                 help="Percentage representation of the average error")
    
    # Optional: Add visual separation
    st.markdown("---")