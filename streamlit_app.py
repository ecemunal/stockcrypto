import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import plotly.graph_objs as go
from datetime import timedelta

# Function to fetch historical data
def fetch_historical_data(ticker, period="1y", interval="1h"):
    stock_data = yf.Ticker(ticker)
    hist_data = stock_data.history(period=period, interval=interval)
    return hist_data

# Function to preprocess data
def preprocess_data(hist_data, scaler, time_step=60):
    scaled_data = scaler.transform(hist_data['Close'].values.reshape(-1,1))
    X = []
    for i in range(len(scaled_data) - time_step - 1):
        a = scaled_data[i:(i + time_step), 0]
        X.append(a)
    return np.array(X)

# Load LSTM model
@st.cache(allow_output_mutation=True)
def load_lstm_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Function to predict the next day
def predict_next_day(model, last_60_days_scaled):
    last_60_days_scaled = last_60_days_scaled.reshape(1, 60, 1)
    predicted_price_scaled = model.predict(last_60_days_scaled)
    return predicted_price_scaled

# Streamlit UI
def main():
    st.title("Stock and Cryptocurrency Price Prediction using LSTM")

    # Dropdown menu for asset selection
    selected_asset = st.selectbox(
        "Choose an asset for prediction:",
        ("META", "BTC-USD", "AAPL"), index=0)  # Default to META

    # Load the appropriate LSTM model based on selection
    if selected_asset == "META":
        lstm_model = load_lstm_model('lstmm_model.h5')
    elif selected_asset == "BTC-USD":
        lstm_model = load_lstm_model('btc_lstm_model.h5')
    elif selected_asset == "AAPL":
        lstm_model = load_lstm_model('aapl_lstm_model.h5')

    if st.button("Predict"):
        with st.spinner('Fetching data and making prediction...'):
            # Fetch historical data for the selected asset
            hist_data = fetch_historical_data(selected_asset)

            # Initialize and fit scaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(hist_data['Close'].values.reshape(-1,1))

            # Preprocess data for prediction
            X = preprocess_data(hist_data, scaler)

            # Predict on historical data
            X = X.reshape(X.shape[0], X.shape[1], 1)
            predicted = lstm_model.predict(X)
            predicted_prices = scaler.inverse_transform(predicted).flatten()

            # Ensure there's enough data for plotting
            if len(predicted_prices) > 0:
                actual_prices = hist_data['Close'].values[-len(predicted_prices):]
                predicted_dates = hist_data.index[-len(predicted_prices):]

                # Create a DataFrame for actual and predicted prices
                prices_df = pd.DataFrame({
                    'Date': predicted_dates,
                    'Actual Price': actual_prices,
                    'Predicted Price': predicted_prices
                })

                # Calculate variance (difference)
                prices_df['Variance'] = prices_df['Actual Price'] - prices_df['Predicted Price']

                # Display the DataFrame in Streamlit
                st.write("Actual vs Predicted Prices with Variance:")
                st.dataframe(prices_df)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=predicted_dates, y=actual_prices, mode='lines', name='Actual Price'))
                fig.add_trace(go.Scatter(x=predicted_dates, y=predicted_prices, mode='lines', name='Predicted Price'))

                # Forecast the next 7 days
                forecast_prices = []
                forecast_dates = [predicted_dates[-1] + timedelta(days=i) for i in range(1, 8)]
                last_60_days_scaled = scaler.transform(hist_data['Close'].values[-60:].reshape(-1, 1))

                for _ in range(7):
                    next_day_price_scaled = predict_next_day(lstm_model, last_60_days_scaled)
                    forecast_prices.append(scaler.inverse_transform(next_day_price_scaled)[0, 0])

                    # Update last 60 days with the new prediction
                    last_60_days_scaled = np.append(last_60_days_scaled[1:], next_day_price_scaled, axis=0)

                fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_prices, mode='lines+markers', name='7-Day Forecast'))

                fig.update_layout(title='Predicted vs Actual Stock Prices with 7-Day Forecast', xaxis_title='Date', yaxis_title='Price')
                
                st.plotly_chart(fig)
            else:
                st.error("Insufficient data for prediction.")

if __name__ == "__main__":
    main()




