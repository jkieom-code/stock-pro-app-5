import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- Configuration ---
st.set_page_config(
    page_title="ProStock | Professional Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Professional UI ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #FAFAFA;
    }
    .metric-card {
        background-color: #262730;
        border: 1px solid #464b5f;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
    }
    /* Hide Streamlit default menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("📈 ProStock Analysis")
st.sidebar.markdown("---")

ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", value=datetime.now())

st.sidebar.subheader("Technical Indicators")
show_sma = st.sidebar.checkbox("SMA (Simple Moving Average)", value=True)
sma_period = st.sidebar.number_input("SMA Period", value=20)
show_ema = st.sidebar.checkbox("EMA (Exp. Moving Average)")
ema_period = st.sidebar.number_input("EMA Period", value=50)
show_bb = st.sidebar.checkbox("Bollinger Bands")
show_rsi = st.sidebar.checkbox("RSI (Relative Strength Index)")

if st.sidebar.button("Run Analysis"):
    st.experimental_rerun()

# --- Helper Functions ---
@st.cache_data
def get_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

@st.cache_data
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info, stock.news
    except Exception as e:
        return None, None

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- Main App Logic ---

# Fetch Data
if ticker:
    data = get_stock_data(ticker, start_date, end_date)
    info, news = get_stock_info(ticker)

    if data is not None and len(data) > 0:
        # Clean data structure (Handle MultiIndex from yfinance)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Calculate Indicators
        data['SMA'] = data['Close'].rolling(window=sma_period).mean()
        data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False).mean()
        data['RSI'] = calculate_rsi(data)
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Upper'] = data['BB_Middle'] + 2 * data['Close'].rolling(window=20).std()
        data['BB_Lower'] = data['BB_Middle'] - 2 * data['Close'].rolling(window=20).std()

        # --- Dashboard Header ---
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        price_change = current_price - prev_close
        pct_change = (price_change / prev_close) * 100
        
        # Safely access dictionary keys with defaults
        market_cap = info.get('marketCap', 'N/A')
        volume = info.get('volume', 'N/A')
        pe_ratio = info.get('trailingPE', 'N/A')

        with col1:
            st.metric(label="Current Price", value=f"${current_price:.2f}", delta=f"{price_change:.2f} ({pct_change:.2f}%)")
        with col2:
            st.metric(label="Market Cap", value=f"{market_cap:,.0f}" if isinstance(market_cap, (int, float)) else market_cap)
        with col3:
            st.metric(label="Volume", value=f"{volume:,.0f}" if isinstance(volume, (int, float)) else volume)
        with col4:
            st.metric(label="P/E Ratio", value=f"{pe_ratio}")

        # --- Tabs ---
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Interactive Chart", "🤖 Price Prediction", "📰 Latest News", "📋 Fundamentals"])

        # Tab 1: Interactive Charts
        with tab1:
            st.subheader(f"{ticker} Price History")
            
            fig = go.Figure()

            # Candlestick
            fig.add_trace(go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'],
                                         name='Price'))

            # Overlays
            if show_sma:
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA'], line=dict(color='orange', width=1), name=f'SMA {sma_period}'))
            if show_ema:
                fig.add_trace(go.Scatter(x=data.index, y=data['EMA'], line=dict(color='cyan', width=1), name=f'EMA {ema_period}'))
            if show_bb:
                fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], line=dict(color='gray', width=1, dash='dash'), name='Upper BB'))
                fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], line=dict(color='gray', width=1, dash='dash'), name='Lower BB'))

            fig.update_layout(height=600, template="plotly_dark", title_text="")
            st.plotly_chart(fig, use_container_width=True)

            if show_rsi:
                st.subheader("Relative Strength Index (RSI)")
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig_rsi.update_layout(height=300, template="plotly_dark")
                st.plotly_chart(fig_rsi, use_container_width=True)

        # Tab 2: Prediction
        with tab2:
            st.subheader("AI Trend Prediction (Linear Regression)")
            st.info("This model uses Linear Regression on recent closing prices to forecast the trend for the next 30 days. Note: This is a statistical projection, not financial advice.")

            # Prepare data for ML
            data_ml = data[['Close']].copy()
            data_ml['Numbers'] = list(range(0, len(data_ml)))
            
            X = np.array(data_ml[['Numbers']])
            y = data_ml['Close'].values
            
            model = LinearRegression().fit(X, y)
            
            # Predict next 30 days
            last_index = data_ml['Numbers'].iloc[-1]
            future_X = np.array([list(range(last_index + 1, last_index + 31))]).reshape(-1, 1)
            future_pred = model.predict(future_X)
            
            # Plotting Prediction
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Historical Price'))
            
            # Create future dates
            last_date = data.index[-1]
            future_dates = [last_date + timedelta(days=x) for x in range(1, 31)]
            
            fig_pred.add_trace(go.Scatter(x=future_dates, y=future_pred, name='Predicted Trend', line=dict(color='red', dash='dot')))
            fig_pred.update_layout(template="plotly_dark", title="30-Day Trend Forecast")
            st.plotly_chart(fig_pred, use_container_width=True)

            col_pred1, col_pred2 = st.columns(2)
            with col_pred1:
                st.metric("Current Price", f"${current_price:.2f}")
            with col_pred2:
                st.metric("Predicted Price (30 Days)", f"${future_pred[-1]:.2f}", delta=f"{future_pred[-1]-current_price:.2f}")

        # Tab 3: News
        with tab3:
            st.subheader(f"Latest News for {ticker}")
            if news:
                for item in news[:10]:
                    # Safe extraction of data with fallbacks
                    title = item.get('title', 'No Title')
                    
                    # Try 'link', 'url', or default to Yahoo Finance
                    link = item.get('link')
                    if not link:
                        link = item.get('url')
                    if not link:
                        link = f"https://finance.yahoo.com/quote/{ticker}/news"
                    
                    publisher = item.get('publisher', 'Unknown')
                    
                    # Safe timestamp conversion
                    try:
                        publish_time = item.get('providerPublishTime')
                        if publish_time:
                            time_str = datetime.fromtimestamp(publish_time).strftime('%Y-%m-%d %H:%M')
                        else:
                            time_str = "Recent"
                    except Exception:
                        time_str = "Recent"

                    st.markdown(f"""
                    <div style='background-color: #262730; padding: 15px; border-radius: 10px; margin-bottom: 10px;'>
                        <a href="{link}" target="_blank" style="text-decoration: none; color: #4DA8DA; font-size: 18px; font-weight: bold;">
                            {title}
                        </a>
                        <p style='color: #BFBFBF; font-size: 14px; margin-top: 5px;'>
                            Publisher: {publisher} | {time_str}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No recent news found.")

        # Tab 4: Fundamentals
        with tab4:
            st.subheader("Company Fundamentals")
            
            # Use columns to display key-value pairs nicely
            f_col1, f_col2 = st.columns(2)
            
            with f_col1:
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                st.write(f"**Country:** {info.get('country', 'N/A')}")
            
            with f_col2:
                st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A')}")
                st.write(f"**Website:** {info.get('website', 'N/A')}")
            
            st.markdown("### Business Summary")
            st.write(info.get('longBusinessSummary', 'N/A'))

    else:
        st.error("Invalid Ticker or No Data Found. Please check the stock symbol.")

else:
    st.info("Please enter a stock ticker in the sidebar to begin analysis.")
