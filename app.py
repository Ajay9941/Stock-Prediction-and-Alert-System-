import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import xgboost as xgb
import datetime
import requests
import streamlit as st

# === Config ===
stock_symbol = "TATAGOLD.NS"
start_date = "2020-01-01"
end_date = datetime.datetime.today().strftime('%Y-%m-%d')

# Telegram Bot Configuration (replace with your actual token/chat ID)
TELEGRAM_TOKEN = "ENTER YOUR TOKEN ID"
TELEGRAM_CHAT_ID = "ENTER YOUR CHAT ID"

# === Streamlit UI ===
st.title("📈 Stock Prediction App (XGBoost)")
st.markdown(f"Analyzing **{stock_symbol}** from {start_date} to {end_date}")

with st.spinner("🔄 Loading and processing data..."):
    # === Download historical stock data ===
    df = yf.download(stock_symbol, start=start_date, end=end_date, auto_adjust=True)

    # Fix MultiIndex columns if needed
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    if isinstance(df['Close'], pd.DataFrame):
        df['Close'] = df['Close'].squeeze()

    # === Technical Indicators ===
    df['MA20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()

    df.dropna(inplace=True)

    # === Features and Labels ===
    features = ['Close', 'MA20', 'RSI', 'BB_High', 'BB_Low']
    X = df[features].values
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)[:-1]
    X = X[:-1]

    # === Train XGBoost Model ===
    model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X, y)

    # === Predict latest signal ===
    latest_data = df[features].iloc[-1:].values
    prediction = model.predict(latest_data)[0]
    signal = "📈 BUY" if prediction == 1 else "📉 SELL"

    # === Get Live Price ===
    try:
        live_price = yf.Ticker(stock_symbol).fast_info['lastPrice']
    except Exception as e:
        live_price = None
        st.warning(f"⚠️ Could not fetch live price: {e}")

# === Show Results ===
st.subheader("📊 Latest Prediction")
st.metric(label="Prediction Signal", value=signal)
if live_price is not None:
    st.metric(label="💰 Live Price", value=f"₹{live_price:.2f}")

st.subheader("📉 Historical Data Snapshot")
st.dataframe(df.tail(10))

st.subheader("📈 Chart (Close & MA20)")
st.line_chart(df[['Close', 'MA20']])

# === Telegram Alert Functions ===
def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, data=payload)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Telegram error: {e}")
        return False

def send_file_to_telegram(file_path):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    files = {"document": open(file_path, "rb")}
    data = {"chat_id": TELEGRAM_CHAT_ID}
    try:
        response = requests.post(url, data=data, files=files)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Telegram file upload error: {e}")
        return False

# === Buttons for Telegram Actions ===
if st.button("📤 Send Signal to Telegram"):
    alert_message = f"*{stock_symbol}* - Signal: *{signal}*"
    if live_price is not None:
        alert_message += f"\n💰 Live Price: ₹{live_price:.2f}"
    if send_telegram_alert(alert_message):
        st.success("✅ Telegram alert sent!")
    else:
        st.error("❌ Failed to send Telegram alert.")

# === Save Data and Send CSV ===
df.to_csv("stock_data_with_indicators.csv", index=True)
st.success("✅ Data saved to 'stock_data_with_indicators.csv'")

if st.button("📁 Send CSV to Telegram"):
    if send_file_to_telegram("stock_data_with_indicators.csv"):
        st.success("✅ CSV file sent to Telegram!")
    else:
        st.error("❌ Failed to send CSV file.")
