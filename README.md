# 📈 Advanced Stock Price Prediction Engine

A real-time, interactive stock price forecasting platform built with **Streamlit**, **TensorFlow/Keras**, **XGBoost**, and **Plotly**. This application combines multiple deep learning sequential architectures, gradient boosting, and ensemble techniques to analyze and predict equity prices for both US and Indian (NSE) stock markets.

---

## 🚀 Key Features

- **Multi-Model Predictive Suite**:
  - **LSTM (Long Short-Term Memory)**: Captures long-term temporal dependencies in time-series data.
  - **GRU (Gated Recurrent Unit)**: Efficient recurrent architecture for fast training with comparable sequential accuracy.
  - **BiLSTM (Bidirectional LSTM)**: Analyzes historical sequences both forward and backward for richer context representation.
  - **CNN-LSTM Hybrid**: Extracts local feature patterns via 1D convolutions before sequential processing with LSTM layers.
  - **XGBoost**: Gradient-boosted decision trees trained on engineered technical features.
  - **Ensemble Model**: Dynamic, weighted combination of individual models for enhanced forecasting robustness.

- **Market Data & Technical Indicators**:
  - Real-time stock data fetched via Yahoo Finance (`yfinance`).
  - Pre-configured quick selectors for popular US (AAPL, TSLA, NVDA, GOOGL) and Indian (TCS, INFY, RELIANCE, HDFCBANK) stocks.
  - Technical indicator engineering: Moving Averages (MA 5, MA 20), Volatility, Daily Return, Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD & Signal Line).

- **Interactive Visualizations & Metrics**:
  - Interactive candlestick charts and price history with volume.
  - Prediction vs. actual comparison curves powered by Plotly.
  - Evaluation metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and Directional Accuracy.
  - Multi-day future forecasting horizon.

---

## 🛠️ Tech Stack

- **Frontend / Dashboard**: [Streamlit](https://streamlit.io/)
- **Data Engineering**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [yfinance](https://pypi.org/project/yfinance/)
- **Deep Learning**: [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/)
- **Machine Learning**: [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/)
- **Visualization**: [Plotly](https://plotly.com/python/)

---

## 💻 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nirajkushwaha98/Advanced-Stock-Price-Prediction.git
   cd Advanced-Stock-Price-Prediction
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv stockenv
   source stockenv/bin/activate
   # On Windows: stockenv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application**:
   ```bash
   streamlit run advanced_stock_prediction.py
   ```

5. **Open the App**:
   Navigate to the local URL displayed in the terminal (typically `http://localhost:8501`).

---

## 📂 Project Structure

```text
Advanced-Stock-Price-Prediction/
├── advanced_stock_prediction.py  # Main Streamlit application and model pipeline
├── requirements.txt              # Required Python libraries
├── .gitignore                    # Git ignore file for Python, env, and OS artifacts
└── README.md                     # Project documentation
```

---

## ⚠️ Disclaimer

This software is for educational, research, and informational purposes only. It is not intended as financial, investment, or trading advice. Past performance does not guarantee future results.
