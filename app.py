import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from prophet import Prophet
import pandas_ta as ta

st.set_page_config(page_title="WallStreetWits Trading Hub", layout="wide")
st.title("🚀 Stock & Crypto Forecasting + Trending App")
st.markdown("**Built for WallStreetWits** – Real-time data, forecasts & momentum scanner")

# Sidebar
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Ticker (e.g. AAPL, TSLA, BTC-USD, ETH-USD)", "BTC-USD")
period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
forecast_days = st.sidebar.slider("Forecast days ahead", 7, 90, 30)

# Fetch data
@st.cache_data(ttl=300)  # cache 5 minutes
def get_data(ticker, period):
    data = yf.download(ticker, period=period, interval="1d")
    data.reset_index(inplace=True)
    data["Date"] = pd.to_datetime(data["Date"])
    # Add technical indicators
    data["SMA_50"] = ta.sma(data["Close"], length=50)
    data["SMA_200"] = ta.sma(data["Close"], length=200)
    data["RSI"] = ta.rsi(data["Close"], length=14)
    data["MACD"] = ta.macd(data["Close"])["MACD_12_26_9"]
    return data

df = get_data(ticker, period)

if df.empty:
    st.error("Invalid ticker or no data. Try AAPL, GOOGL, BTC-USD, SOL-USD")
    st.stop()

# Current price & stats
current_price = df["Close"].iloc[-1]
change = ((current_price - df["Close"].iloc[-2]) / df["Close"].iloc[-2]) * 100
st.metric(f"**{ticker}**", f"${current_price:,.2f}", f"{change:+.2f}%")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Chart + Indicators", "🔮 Price Forecast", "🔥 Trending Scanner", "📈 Backtest"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_50"], name="SMA 50", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_200"], name="SMA 200", line=dict(color="blue")))
    fig.update_layout(title=f"{ticker} Price & Indicators", xaxis_title="Date", yaxis_title="Price (USD)", height=600, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RSI")
        st.line_chart(df.set_index("Date")["RSI"])
    with col2:
        st.subheader("MACD")
        st.line_chart(df.set_index("Date")["MACD"])

with tab2:
    st.subheader(f"{forecast_days}-Day Price Forecast using Prophet")
    # Prepare data for Prophet
    prophet_df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(prophet_df)
    
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    
    # Plot forecast
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["y"], name="Actual", line=dict(color="blue")))
    fig_forecast.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast", line=dict(color="green")))
    fig_forecast.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], name="Lower Bound", line=dict(color="red", dash="dot")))
    fig_forecast.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], name="Upper Bound", line=dict(color="red", dash="dot")))
    fig_forecast.update_layout(title="Price Forecast", height=500, template="plotly_dark")
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.success(f"Predicted price in {forecast_days} days: **${forecast['yhat'].iloc[-1]:,.2f}**")

with tab3:
    st.subheader("🔥 Trending Scanner (Top Movers)")
    # Quick scanner for popular assets
    popular = ["AAPL", "TSLA", "NVDA", "GOOGL", "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
    movers = []
    for t in popular:
        try:
            d = yf.download(t, period="5d")["Close"]
            pct = ((d.iloc[-1] - d.iloc[-2]) / d.iloc[-2]) * 100
            movers.append({"Ticker": t, "Price": d.iloc[-1], "Change %": pct})
        except:
            pass
    movers_df = pd.DataFrame(movers).sort_values("Change %", ascending=False)
    st.dataframe(movers_df.style.format({"Price": "${:,.2f}", "Change %": "{:+.2f}%"}), use_container_width=True)

with tab4:
    st.info("Backtesting coming in next version (simple strategy tester with SMA crossover). Want it sooner?")

st.caption("Data from Yahoo Finance • Forecasts are for illustration only • Not financial advice")