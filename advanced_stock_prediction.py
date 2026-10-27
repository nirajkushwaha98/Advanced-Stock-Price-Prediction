import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

st.set_page_config(
    page_title="Advanced Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 2px;
}
.subtitle {
    text-align: center;
    color: #888;
    margin-bottom: 20px;
    font-size: 1rem;
}
.metric-container {
    background: rgba(255, 255, 255, 0.05);
    padding: 12px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>📈 Advanced Stock Price Prediction Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deep Learning (LSTM, GRU, BiLSTM, CNN-LSTM) + XGBoost + Ensemble AI</div>", unsafe_allow_html=True)

FEATURE_COLS = [
    "MA_5", "MA_20", "Volatility",
    "Daily_Return", "RSI", "MACD", "Signal_Line"
]
ALL_DL_MODELS = ["LSTM", "GRU", "BiLSTM", "CNN-LSTM"]

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Configuration")

def set_ticker(ticker_name):
    st.session_state["stock_ticker_input"] = ticker_name

if "stock_ticker_input" not in st.session_state:
    st.session_state["stock_ticker_input"] = "AAPL"

stock_symbol = st.sidebar.text_input(
    "Enter Stock Symbol",
    key="stock_ticker_input",
    help="US: AAPL, MSFT, TSLA, NVDA | India: TCS.NS, INFY.NS, RELIANCE.NS (or type TCS directly)"
).strip().upper()

# Quick Symbol selector
st.sidebar.caption("Popular US Tickers:")
quick_us = st.sidebar.columns(4)
quick_us[0].button("AAPL", on_click=set_ticker, args=("AAPL",))
quick_us[1].button("TSLA", on_click=set_ticker, args=("TSLA",))
quick_us[2].button("NVDA", on_click=set_ticker, args=("NVDA",))
quick_us[3].button("GOOGL", on_click=set_ticker, args=("GOOGL",))

st.sidebar.caption("Popular Indian Tickers (NSE):")
quick_in = st.sidebar.columns(4)
quick_in[0].button("TCS", on_click=set_ticker, args=("TCS.NS",))
quick_in[1].button("INFY", on_click=set_ticker, args=("INFY.NS",))
quick_in[2].button("RELIANCE", on_click=set_ticker, args=("RELIANCE.NS",))
quick_in[3].button("HDFC", on_click=set_ticker, args=("HDFCBANK.NS",))

st.sidebar.markdown("---")

days_back = st.sidebar.number_input(
    "Days of History", min_value=120, max_value=2500,
    value=500, step=50
)

lookback = st.sidebar.slider(
    "Lookback Window (days)", min_value=10, max_value=90,
    value=30, step=5
)

test_split = st.sidebar.slider(
    "Test Split %", min_value=10, max_value=30,
    value=20, step=5
)

st.sidebar.subheader("🤖 Model Selection")
selected_models = st.sidebar.multiselect(
    "Models to Train",
    options=["LSTM", "GRU", "BiLSTM", "CNN-LSTM", "XGBoost"],
    default=["LSTM", "XGBoost"],
    help="Tip: Select 1 or 2 models for fast results (10-20s), or all models for full benchmark."
)

epochs = st.sidebar.slider(
    "Deep Learning Epochs", min_value=5, max_value=60,
    value=15, step=5,
    help="Lower epochs (10-15) train much faster with early stopping."
)

xgb_rounds = st.sidebar.slider(
    "XGBoost Rounds", min_value=50, max_value=400,
    value=150, step=25
)

batch_size = st.sidebar.selectbox(
    "Batch Size", [16, 32, 64], index=1
)

ensemble_mode = st.sidebar.selectbox(
    "Ensemble Method",
    ["Validation-Weighted", "Average"]
)

use_demo_fallback = st.sidebar.checkbox(
    "Use demo data if live data fails", value=True
)

run_prediction = st.sidebar.button(
    "🚀 Train Models & Predict",
    type="primary"
)

# ============================================================
# DATA LOADING & NORMALIZATION
# ============================================================
def normalize_price_data(data):
    if data is None or data.empty:
        return pd.DataFrame()

    df = data.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in df.columns
        ]

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Close" in df.columns:
        df = df.dropna(subset=["Close"])

    return df.sort_index()


@st.cache_data(ttl=3600)
def generate_demo_data(days):
    trading_days = max(days, 180)
    dates = pd.bdate_range(
        end=datetime.now().date(),
        periods=trading_days
    )

    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.018, size=len(dates))
    close = 150 * np.exp(np.cumsum(returns))

    open_price = close * (1 + rng.normal(0, 0.003, len(dates)))
    high = np.maximum(open_price, close) * (
        1 + rng.uniform(0, 0.01, len(dates))
    )
    low = np.minimum(open_price, close) * (
        1 - rng.uniform(0, 0.01, len(dates))
    )
    volume = rng.integers(1_000_000, 8_000_000, len(dates))

    return pd.DataFrame({
        "Open": open_price,
        "High": high,
        "Low": low,
        "Close": close,
        "Adj Close": close,
        "Volume": volume
    }, index=dates)


@st.cache_data(ttl=3600)
def load_yahoo_chart_api_data(symbol, days):
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=int(days) + 7)

    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {
        "period1": int(start_dt.timestamp()),
        "period2": int(end_dt.timestamp()),
        "interval": "1d",
        "events": "div,splits",
        "includePrePost": "false"
    }

    try:
        response = requests.get(
            url, headers=headers, params=params, timeout=15
        )

        if response.status_code != 200:
            return pd.DataFrame()

        payload = response.json()
        result = (payload.get("chart", {}).get("result") or [None])[0]

        if result is None:
            return pd.DataFrame()

        timestamps = result.get("timestamp") or []
        quote = (result.get("indicators", {}).get("quote") or [{}])[0]
        adj = (result.get("indicators", {}).get("adjclose") or [{}])[0]

        if not timestamps or not quote:
            return pd.DataFrame()

        df = pd.DataFrame(
            {
                "Open": quote.get("open", []),
                "High": quote.get("high", []),
                "Low": quote.get("low", []),
                "Close": quote.get("close", []),
                "Volume": quote.get("volume", []),
                "Adj Close": adj.get(
                    "adjclose", quote.get("close", [])
                )
            },
            index=pd.to_datetime(timestamps, unit="s")
        ).sort_index()

        for col in [
            "Open", "High", "Low", "Close", "Adj Close", "Volume"
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.dropna(subset=["Close"])

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_raw_ticker(symbol, days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    errors = []

    try:
        data = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            progress=False,
            interval="1d",
            auto_adjust=False,
            threads=False
        )
        data = normalize_price_data(data)
        if not data.empty:
            return data, None
    except Exception as e:
        errors.append(str(e))

    try:
        data = yf.Ticker(symbol).history(
            period=f"{max(days, 120)}d",
            interval="1d",
            auto_adjust=False
        )
        data = normalize_price_data(data)
        if not data.empty:
            return data, None
    except Exception as e:
        errors.append(str(e))

    data = load_yahoo_chart_api_data(symbol, days)
    if not data.empty:
        return data, None

    if not errors:
        errors.append("No market data returned.")

    return pd.DataFrame(), " | ".join(errors)


@st.cache_data(ttl=3600)
def load_stock_data(symbol, days):
    if not symbol or not str(symbol).strip():
        return pd.DataFrame(), "", "Please enter a valid stock symbol."
    symbol = str(symbol).strip().upper()

    # 1. Try exact symbol first
    data, error = fetch_raw_ticker(symbol, days)
    if not data.empty:
        return data, symbol, None

    # 2. If symbol has no dot (e.g. TCS, INFY, RELIANCE), try Indian suffixes .NS (NSE) and .BO (BSE)
    if "." not in symbol:
        for suffix in [".NS", ".BO"]:
            alt_symbol = f"{symbol}{suffix}"
            alt_data, _ = fetch_raw_ticker(alt_symbol, days)
            if not alt_data.empty:
                return alt_data, alt_symbol, None

    return pd.DataFrame(), symbol, error


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def preprocess_data(data):
    if data is None or data.empty or "Close" not in data.columns:
        return pd.DataFrame()

    df = data[["Close"]].copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.ffill().bfill()

    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["Volatility"] = df["Close"].rolling(20).std()
    df["Daily_Return"] = df["Close"].pct_change()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    exp12 = df["Close"].ewm(span=12, adjust=False).mean()
    exp26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = exp12 - exp26
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


# ============================================================
# DEEP LEARNING MODEL BUILDERS
# ============================================================
def create_sequences(values, lookback):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i])
        y.append(values[i])
    return np.array(X), np.array(y)


def build_lstm(lookback):
    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def build_gru(lookback):
    model = Sequential([
        Input(shape=(lookback, 1)),
        GRU(64, return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def build_bilstm(lookback):
    model = Sequential([
        Input(shape=(lookback, 1)),
        Bidirectional(LSTM(48, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(24)),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def build_cnn_lstm(lookback):
    model = Sequential([
        Input(shape=(lookback, 1)),
        Conv1D(filters=64, kernel_size=3, activation="relu"),
        MaxPooling1D(pool_size=2),
        LSTM(48),
        Dropout(0.2),
        Dense(24, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def build_model(name, lookback):
    if name == "LSTM":
        return build_lstm(lookback)
    if name == "GRU":
        return build_gru(lookback)
    if name == "BiLSTM":
        return build_bilstm(lookback)
    if name == "CNN-LSTM":
        return build_cnn_lstm(lookback)
    raise ValueError(f"Unknown model: {name}")


def train_deep_model(name, train_values, test_values, lookback, epochs, batch):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_values.reshape(-1, 1))

    X_train, y_train = create_sequences(train_scaled, lookback)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    model = build_model(name, lookback)
    early_stop = EarlyStopping(
        monitor="loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch,
        verbose=0,
        callbacks=[early_stop]
    )

    test_input = np.concatenate([train_values[-lookback:], test_values]).reshape(-1, 1)
    test_scaled = scaler.transform(test_input)

    X_test, y_test = create_sequences(test_scaled, lookback)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    pred_scaled = model.predict(X_test, verbose=0)
    pred = scaler.inverse_transform(pred_scaled).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    return model, scaler, pred, actual


def recursive_forecast(model, scaler, full_values, lookback, days):
    scaled = scaler.transform(full_values.reshape(-1, 1))
    sequence = scaled[-lookback:].copy()
    predictions = []

    for _ in range(days):
        pred_scaled = model.predict(
            sequence.reshape(1, lookback, 1),
            verbose=0
        )[0, 0]
        predictions.append(pred_scaled)
        sequence = np.vstack([sequence[1:], [[pred_scaled]]])

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()


# ============================================================
# XGBOOST MODEL BUILDER
# ============================================================
def prepare_xgb_data(df, lookback):
    X, y = [], []
    feature_values = df[FEATURE_COLS].values
    close_values = df["Close"].values

    for i in range(lookback, len(df)):
        X.append(feature_values[i - lookback:i].flatten())
        y.append(close_values[i])

    return np.array(X), np.array(y)


def train_xgb(df, lookback, test_split, rounds):
    X, y = prepare_xgb_data(df, lookback)
    split = int(len(X) * (1 - test_split / 100))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = xgb.XGBRegressor(
        n_estimators=rounds,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train, verbose=False)
    pred = model.predict(X_test)
    return model, pred, y_test, X, y


# ============================================================
# METRICS & CONFIDENCE CALCULATION
# ============================================================
def calculate_metrics(actual, pred):
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    safe_actual = np.where(actual == 0, 1e-8, actual)
    mape = np.mean(np.abs((actual - pred) / safe_actual)) * 100

    # Directional Hit Rate (% of correct UP/DOWN price movement calls)
    if len(actual) > 1:
        actual_diff = np.diff(actual)
        pred_diff = np.diff(pred)
        dir_acc = float(np.mean((actual_diff * pred_diff) > 0) * 100)
    else:
        dir_acc = 50.0

    # Price Precision Accuracy (100% - MAPE)
    price_acc = max(0.0, min(100.0, 100.0 - mape))

    # Overall Model Confidence Score %:
    # 60% Weight on Price Precision + 40% Weight on Directional Trend Accuracy
    confidence_pct = max(0.0, min(100.0, 0.6 * price_acc + 0.4 * dir_acc))

    return mae, rmse, mape, dir_acc, price_acc, confidence_pct


# ============================================================
# MAIN APPLICATION FLOW
# ============================================================
if not stock_symbol:
    st.info("👈 Enter a stock symbol in the sidebar to begin.")
    st.stop()

# 1. Immediate Data Fetch (Instant Display)
with st.spinner(f"📥 Loading market data for '{stock_symbol}'..."):
    data, resolved_symbol, error = load_stock_data(stock_symbol, days_back)

if not data.empty and "Close" in data.columns:
    data = data.dropna(subset=["Close"])

# Auto-resolution notification for Indian tickers
if resolved_symbol != stock_symbol:
    st.info(f"💡 Auto-resolved **'{stock_symbol}'** to **'{resolved_symbol}'** (National Stock Exchange of India).")
    display_symbol = resolved_symbol
else:
    display_symbol = stock_symbol

# Determine currency symbol
currency_sym = "₹" if (display_symbol.endswith(".NS") or display_symbol.endswith(".BO")) else "$"

if data.empty and use_demo_fallback:
    data = generate_demo_data(days_back)
    st.warning("⚠️ Live market data unavailable. Generated synthetic demo data for demonstration.")

if data.empty:
    st.error(f"❌ Could not load stock data for '{stock_symbol}'.")
    st.markdown("""
    **💡 Tips for Stock Tickers on Yahoo Finance:**
    - **US Stocks:** Use symbol directly (e.g., `AAPL`, `MSFT`, `TSLA`, `GOOGL`, `NVDA`)
    - **Indian Stocks (NSE):** Add `.NS` (e.g., `TCS.NS`, `INFY.NS`, `RELIANCE.NS`, `HDFCBANK.NS`, `TATAMOTORS.NS`)
    - **Indian Stocks (BSE):** Add `.BO` (e.g., `TCS.BO`, `INFY.BO`, `RELIANCE.BO`)
    """)
    if error:
        st.caption(f"Details: {error}")
    st.stop()

if len(data) <= lookback + 50:
    st.error(f"❌ Not enough data rows ({len(data)} received). Please increase 'Days of History' or reduce 'Lookback Window'.")
    st.stop()

# Preprocess technical indicators
df = preprocess_data(data)
if len(df) <= lookback + 30:
    st.error("❌ Not enough data remaining after indicator computation. Increase history days.")
    st.stop()

# Compute snapshot stats using clean verified data
current_price = float(df["Close"].iloc[-1])
previous_price = float(df["Close"].iloc[-2]) if len(df) > 1 else current_price
change = current_price - previous_price
change_pct = (change / previous_price * 100) if previous_price != 0 else 0
high_52 = float(df["Close"].tail(252).max())
low_52 = float(df["Close"].tail(252).min())
volatility_20 = float(df["Volatility"].iloc[-1])
latest_rsi = float(df["RSI"].iloc[-1])

# Header Metrics Overview
st.subheader(f"📊 Market Overview: {display_symbol}")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Current Price", f"{currency_sym}{current_price:.2f}", f"{change:+.2f}")
m2.metric("Daily Change", f"{change_pct:+.2f}%")
m3.metric("52-Week High", f"{currency_sym}{high_52:.2f}")
m4.metric("52-Week Low", f"{currency_sym}{low_52:.2f}")
m5.metric("20D Volatility", f"{volatility_20:.2f}")
m6.metric("RSI (14)", f"{latest_rsi:.1f}")

# Tabs for Market Data
tab1, tab2, tab3 = st.tabs(["🕯️ Candlestick & Price Action", "📈 Technical Indicators", "📋 Recent Data"])

with tab1:
    fig_candle = go.Figure()
    if all(col in data.columns for col in ["Open", "High", "Low", "Close"]):
        fig_candle.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="OHLC"
        ))
    else:
        fig_candle.add_trace(go.Scatter(
            x=data.index,
            y=data["Close"],
            name="Close Price",
            mode="lines"
        ))
    
    # Add Moving Averages
    if "MA_5" in df.columns:
        fig_candle.add_trace(go.Scatter(x=df.index, y=df["MA_5"], name="5-Day MA", line=dict(color="orange", width=1.5)))
    if "MA_20" in df.columns:
        fig_candle.add_trace(go.Scatter(x=df.index, y=df["MA_20"], name="20-Day MA", line=dict(color="cyan", width=1.5)))

    fig_candle.update_layout(
        title=f"{display_symbol} Historical Price Action & Moving Averages",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_sym})",
        hovermode="x unified",
        height=480,
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig_candle)

with tab2:
    col_ind1, col_ind2 = st.columns(2)
    with col_ind1:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI (14)", line=dict(color="#b388ff")))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig_rsi.update_layout(
            title="Relative Strength Index (RSI)",
            yaxis=dict(range=[0, 100]),
            height=300,
            template="plotly_dark"
        )
        st.plotly_chart(fig_rsi)

    with col_ind2:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#00e5ff")))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["Signal_Line"], name="Signal Line", line=dict(color="#ff9100")))
        fig_macd.update_layout(
            title="Moving Average Convergence Divergence (MACD)",
            height=300,
            template="plotly_dark"
        )
        st.plotly_chart(fig_macd)

with tab3:
    st.dataframe(data.tail(20).sort_index(ascending=False))

st.divider()

# ============================================================
# PREDICTION & MODEL TRAINING SECTION
# ============================================================
st.subheader("🎯 Machine Learning & Deep Learning Forecasting")

if not selected_models:
    st.warning("⚠️ Please select at least one model in the sidebar to train.")
    st.stop()

# Check if we should execute training or if results already exist in session_state
should_train = run_prediction
has_results = (
    "prediction_results" in st.session_state and 
    st.session_state.get("predicted_symbol") == display_symbol and
    set(st.session_state.get("trained_models", [])) == set(selected_models)
)

if not should_train and not has_results:
    st.info(
        f"👉 **Ready to train models for {display_symbol}!**\n\n"
        f"Selected Models: **{', '.join(selected_models)}** | Lookback: **{lookback} days** | Epochs: **{epochs}**\n\n"
        f"Click the **'🚀 Train Models & Predict'** button in the sidebar to execute training and generate forecasts."
    )
    st.stop()

if should_train:
    train_size = int(len(df) * (1 - test_split / 100))
    train_values = df["Close"].values[:train_size]
    test_values = df["Close"].values[train_size:]

    if len(train_values) <= lookback:
        st.error("❌ Training dataset is too small. Increase historical days.")
        st.stop()

    models = {}
    scalers = {}
    predictions = {}
    actuals = {}
    xgb_model = None
    xgb_pred = None
    xgb_actual = None

    with st.status(f"🤖 Training AI models for {display_symbol}...", expanded=True) as status:
        # Train Deep Learning Models
        selected_dl = [m for m in selected_models if m in ALL_DL_MODELS]
        for name in selected_dl:
            st.write(f"🧠 Training **{name}** Neural Network ({epochs} epochs)...")
            m, sc, p, a = train_deep_model(name, train_values, test_values, lookback, epochs, batch_size)
            models[name] = m
            scalers[name] = sc
            predictions[name] = p
            actuals[name] = a

        # Train XGBoost if selected
        if "XGBoost" in selected_models:
            st.write(f"🚀 Training **XGBoost Regressor** ({xgb_rounds} estimators)...")
            xgb_model, xgb_pred, xgb_actual, X_xgb, y_xgb = train_xgb(df, lookback, test_split, xgb_rounds)

        st.write("📊 Calculating performance metrics & assembling Ensemble...")

        # Align lengths of test predictions
        all_preds = {}
        lens = [len(predictions[m]) for m in selected_dl]
        if "XGBoost" in selected_models:
            lens.append(len(xgb_pred))

        min_len = min(lens)
        for name in selected_dl:
            all_preds[name] = predictions[name][-min_len:]
        if "XGBoost" in selected_models:
            all_preds["XGBoost"] = xgb_pred[-min_len:]

        # Common ground truth
        if "XGBoost" in selected_models:
            common_actual = xgb_actual[-min_len:]
        else:
            first_dl = selected_dl[0]
            common_actual = actuals[first_dl][-min_len:]

        # Calculate metrics
        metric_rows = []
        for name, pred in all_preds.items():
            mae, rmse, mape, dir_acc, price_acc, conf_pct = calculate_metrics(common_actual, pred)
            metric_rows.append({
                "Model": name,
                "Confidence (%)": conf_pct,
                "Price Accuracy (%)": price_acc,
                "Directional Trend (%)": dir_acc,
                "MAPE (%)": mape,
                "MAE": mae,
                "RMSE": rmse
            })

        metrics_df = pd.DataFrame(metric_rows)

        # Ensemble logic
        if len(all_preds) > 1:
            pred_matrix = np.vstack([all_preds[name] for name in all_preds])
            if ensemble_mode == "Average":
                ensemble_pred = np.mean(pred_matrix, axis=0)
            else:
                raw_weights = [1 / max(row["MAPE (%)"], 0.01) for _, row in metrics_df.iterrows()]
                weights = np.array(raw_weights) / sum(raw_weights)
                ensemble_pred = np.average(pred_matrix, axis=0, weights=weights)

            ens_mae, ens_rmse, ens_mape, ens_dir, ens_price, ens_conf = calculate_metrics(common_actual, ensemble_pred)
            metrics_df = pd.concat([
                metrics_df,
                pd.DataFrame([{
                    "Model": "Ensemble",
                    "Confidence (%)": ens_conf,
                    "Price Accuracy (%)": ens_price,
                    "Directional Trend (%)": ens_dir,
                    "MAPE (%)": ens_mape,
                    "MAE": ens_mae,
                    "RMSE": ens_rmse
                }])
            ], ignore_index=True)
        else:
            ensemble_pred = list(all_preds.values())[0]
            ens_conf = metric_rows[0]["Confidence (%)"]

        # 7-day Future Forecast
        st.write("🔮 Generating recursive 7-day future price forecast...")
        future_days = 7
        future_predictions = {}
        full_close = df["Close"].values

        for name in selected_dl:
            future_predictions[name] = recursive_forecast(models[name], scalers[name], full_close, lookback, future_days)

        if "XGBoost" in selected_models:
            temp_df = df.copy()
            xgb_future = []
            for _ in range(future_days):
                latest = temp_df.iloc[-1]["Close"]
                last_window = temp_df[FEATURE_COLS].tail(lookback).values.flatten()
                if len(last_window) == lookback * len(FEATURE_COLS):
                    next_pred = float(xgb_model.predict(last_window.reshape(1, -1))[0])
                else:
                    next_pred = float(latest)
                xgb_future.append(next_pred)
                new_row = temp_df.iloc[-1].copy()
                new_row["Close"] = next_pred
                temp_df = pd.concat([temp_df, pd.DataFrame([new_row], index=[temp_df.index[-1] + pd.Timedelta(days=1)])])
            future_predictions["XGBoost"] = np.array(xgb_future)

        future_matrix = np.vstack([future_predictions[name] for name in all_preds])
        if len(all_preds) > 1 and ensemble_mode != "Average":
            future_ensemble = np.average(future_matrix, axis=0, weights=weights)
        else:
            future_ensemble = np.mean(future_matrix, axis=0)

        uncertainty = np.std(future_matrix, axis=0) if len(all_preds) > 1 else np.zeros(future_days)
        upper_band = future_ensemble + 1.96 * uncertainty
        lower_band = future_ensemble - 1.96 * uncertainty

        # Compute Day-by-Day Forecast Confidence for future 7 days
        future_confidence = []
        for t in range(future_days):
            horizon_decay = 1.0 - 0.018 * t
            if len(all_preds) > 1 and future_ensemble[t] > 0:
                rel_spread = float(uncertainty[t] / future_ensemble[t])
                spread_factor = max(0.80, 1.0 - 2.5 * rel_spread)
            else:
                spread_factor = 1.0
            day_conf = round(float(np.clip(ens_conf * horizon_decay * spread_factor, 50.0, 99.5)), 1)
            future_confidence.append(f"{day_conf:.1f}%")

        moves = []
        last_p = current_price
        for fp in future_ensemble:
            mv = ((fp - last_p) / last_p) * 100
            moves.append(f"{mv:+.2f}%")
            last_p = fp

        future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=future_days, freq="B")
        forecast_df = pd.DataFrame({
            "Date": future_dates.strftime("%Y-%m-%d"),
            "Ensemble Forecast": [f"{currency_sym}{p:.2f}" for p in future_ensemble],
            "Expected Move": moves,
            "Forecast Confidence (%)": future_confidence,
            "Expected Range (±1.96σ)": [f"{currency_sym}{l:.2f} — {currency_sym}{u:.2f}" for l, u in zip(lower_band, upper_band)]
        })
        for name in all_preds:
            forecast_df[f"{name} ({currency_sym})"] = [f"{currency_sym}{p:.2f}" for p in future_predictions[name]]

        # Store in session state
        st.session_state["predicted_symbol"] = display_symbol
        st.session_state["trained_models"] = selected_models
        st.session_state["prediction_results"] = {
            "metrics_df": metrics_df,
            "all_preds": all_preds,
            "common_actual": common_actual,
            "ensemble_pred": ensemble_pred,
            "test_dates": df.index[-min_len:],
            "xgb_model": xgb_model,
            "forecast_df": forecast_df,
            "future_dates": future_dates,
            "future_ensemble": future_ensemble,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "lookback": lookback,
            "currency_sym": currency_sym
        }
        status.update(label="✅ Training Complete & Forecast Generated!", state="complete", expanded=False)

# Retrieve from session state
results = st.session_state["prediction_results"]
metrics_df = results["metrics_df"]
all_preds = results["all_preds"]
common_actual = results["common_actual"]
ensemble_pred = results["ensemble_pred"]
test_dates = results["test_dates"]
xgb_model = results["xgb_model"]
forecast_df = results["forecast_df"]
future_dates = results["future_dates"]
future_ensemble = results["future_ensemble"]
upper_band = results["upper_band"]
lower_band = results["lower_band"]
res_curr = results.get("currency_sym", currency_sym)

# ============================================================
# DISPLAY RESULTS
# ============================================================

# Model Metrics
st.markdown("#### 🏆 Model Performance Benchmark (Test Set)")
disp_metrics = metrics_df[["Model", "MAE", "RMSE", "MAPE (%)", "Directional Trend (%)"]].copy()
disp_metrics["MAPE (%)"] = disp_metrics["MAPE (%)"].round(2)
disp_metrics["Directional Trend (%)"] = disp_metrics["Directional Trend (%)"].round(2)
disp_metrics["MAE"] = disp_metrics["MAE"].round(3)
disp_metrics["RMSE"] = disp_metrics["RMSE"].round(3)
st.dataframe(disp_metrics, hide_index=True)

st.divider()

# Test Predictions Comparison Chart
st.markdown("#### 📈 Model Predictions vs Actual Prices (Test Evaluation)")
fig_comp = go.Figure()
fig_comp.add_trace(go.Scatter(x=test_dates, y=common_actual, name="Actual Close", line=dict(color="#ffffff", width=2.5)))
colors = ["#00e5ff", "#76ff03", "#ff9100", "#d500f9", "#ffd600"]
for idx, (name, p) in enumerate(all_preds.items()):
    fig_comp.add_trace(go.Scatter(
        x=test_dates, y=p, name=name,
        line=dict(color=colors[idx % len(colors)], width=1.5, dash="dot")
    ))

if len(all_preds) > 1:
    fig_comp.add_trace(go.Scatter(
        x=test_dates, y=ensemble_pred, name="Ensemble",
        line=dict(color="#ff1744", width=3)
    ))

fig_comp.update_layout(
    title=f"{display_symbol} — Out-of-Sample Test Evaluation",
    xaxis_title="Date",
    yaxis_title=f"Price ({res_curr})",
    hovermode="x unified",
    height=500,
    template="plotly_dark"
)
st.plotly_chart(fig_comp)

# Error Distribution & Feature Importance
c_err, c_imp = st.columns(2)
with c_err:
    st.markdown("#### 📉 Absolute Error Distribution")
    error_df = pd.DataFrame({name: np.abs(common_actual - p) for name, p in all_preds.items()})
    if len(all_preds) > 1:
        error_df["Ensemble"] = np.abs(common_actual - ensemble_pred)
    fig_box = px.box(error_df, y=error_df.columns, title=f"Prediction Error Spread ({res_curr})")
    fig_box.update_layout(height=420, template="plotly_dark")
    st.plotly_chart(fig_box)

with c_imp:
    st.markdown("#### 🔥 Feature Importance")
    if xgb_model is not None:
        base_feature_names = []
        for lag in range(results["lookback"], 0, -1):
            for feature in FEATURE_COLS:
                base_feature_names.append(f"{feature}_t-{lag}")
        importance = xgb_model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": base_feature_names[:len(importance)],
            "Importance": importance
        }).sort_values("Importance", ascending=False).head(15)

        fig_imp = px.bar(
            importance_df.sort_values("Importance"),
            x="Importance", y="Feature", orientation="h",
            title="Top 15 XGBoost Features"
        )
        fig_imp.update_layout(height=420, template="plotly_dark")
        st.plotly_chart(fig_imp)
    else:
        st.info("Train XGBoost to inspect technical feature importances.")

st.divider()

# Future Forecast
st.markdown("#### 🔮 Next 7 Trading Days Forecast")
st.dataframe(forecast_df, hide_index=True)

fig_forecast = go.Figure()
history_days = min(60, len(data))
fig_forecast.add_trace(go.Scatter(
    x=data.index[-history_days:],
    y=data["Close"].iloc[-history_days:],
    name="Historical Close",
    line=dict(color="#ffffff", width=2)
))
fig_forecast.add_trace(go.Scatter(
    x=future_dates,
    y=future_ensemble,
    name="Ensemble Forecast",
    mode="lines+markers",
    line=dict(color="#00e676", width=3)
))

if len(all_preds) > 1:
    fig_forecast.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(upper_band) + list(lower_band[::-1]),
        fill="toself",
        fillcolor="rgba(0, 230, 118, 0.15)",
        name="Model Uncertainty Range",
        line=dict(width=0),
        hoverinfo="skip"
    ))

fig_forecast.update_layout(
    title=f"{display_symbol} — 7-Day Recursive Price Forecast with Model Spread",
    xaxis_title="Date",
    yaxis_title=f"Price ({res_curr})",
    hovermode="x unified",
    height=500,
    template="plotly_dark"
)
st.plotly_chart(fig_forecast)

# ============================================================
# FINAL MODEL CONFIDENCE (ONLY FOR TOP MODEL - LAST ME NICHE)
# ============================================================
st.divider()
st.subheader("🎯 Top Model Confidence Analysis")

top_model_row = metrics_df.loc[metrics_df["Confidence (%)"].idxmax()]
top_model_name = str(top_model_row["Model"])
top_conf = float(top_model_row["Confidence (%)"])
top_price_acc = float(top_model_row["Price Accuracy (%)"])
top_dir_acc = float(top_model_row["Directional Trend (%)"])
top_mape = float(top_model_row["MAPE (%)"])

if top_conf >= 90:
    conf_badge = "🟢 High Confidence"
    badge_bg = "rgba(0, 230, 118, 0.2)"
    badge_color = "#00e676"
    conf_desc = f"{top_model_name} demonstrated the highest predictive precision and consistent directional accuracy."
elif top_conf >= 80:
    conf_badge = "🟡 Moderate Confidence"
    badge_bg = "rgba(255, 214, 0, 0.2)"
    badge_color = "#ffd600"
    conf_desc = f"{top_model_name} leads performance with reliable short-term trend alignment under normal volatility."
else:
    conf_badge = "🟠 Cautious (Elevated Volatility)"
    badge_bg = "rgba(255, 145, 0, 0.2)"
    badge_color = "#ff9100"
    conf_desc = f"{top_model_name} achieved the best score amid wider price swings in recent market history."

st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.02));
            border: 1px solid rgba(0, 230, 118, 0.35); border-radius: 12px; padding: 22px 28px; margin-bottom: 25px;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
        <div>
            <div style="font-size: 0.95rem; color: #aaa; text-transform: uppercase; letter-spacing: 0.5px;">
                🏆 Top Performing Model: <b style="color: #fff; font-size: 1.15rem;">{top_model_name}</b>
            </div>
            <div style="display: flex; align-items: baseline; gap: 12px; margin-top: 6px;">
                <span style="font-size: 2.8rem; font-weight: 800; color: {badge_color};">{top_conf:.1f}%</span>
                <span style="background: {badge_bg}; color: {badge_color}; padding: 4px 14px; border-radius: 20px; font-size: 0.9rem; font-weight: 600;">{conf_badge}</span>
            </div>
            <div style="color: #888; font-size: 0.88rem; margin-top: 6px;">{conf_desc}</div>
        </div>
        <div style="background: rgba(0,0,0,0.35); padding: 14px 22px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">
            <div style="margin-bottom: 6px; font-size: 0.95rem;">📈 Price Precision: <b style="color: #00e5ff;">{top_price_acc:.1f}%</b></div>
            <div style="margin-bottom: 6px; font-size: 0.95rem;">🧭 Trend Direction Accuracy: <b style="color: #76ff03;">{top_dir_acc:.1f}%</b></div>
            <div style="font-size: 0.9rem; color: #bbb;">📉 Test Error (MAPE): <b style="color: #ff9100;">{top_mape:.2f}%</b></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Summary Cards
st.markdown("---")
forecast_change = (((future_ensemble[-1] - current_price) / current_price) * 100) if current_price != 0 and not np.isnan(current_price) else 0.0

s1, s2, s3 = st.columns(3)
s1.metric("Top Model Confidence", f"{top_conf:.1f}%", f"{top_model_name} (Top Model)")
s2.metric("Projected 7-Day Change", f"{forecast_change:+.2f}%", f"{res_curr}{future_ensemble[-1]:.2f}")
s3.metric("Current Reference Price", f"{res_curr}{current_price:.2f}")

st.info(
    "⚠️ **Disclaimer:** This application is for educational and quantitative demonstration purposes only. "
    "Stock prices are influenced by unforeseen market events, company news, and global factors not captured by technical sequence models."
)
