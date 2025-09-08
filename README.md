# 🚀 AI Stock Market Analyzer

**AI Stock Market Analyzer** is an interactive dashboard built with **Python** and **Streamlit** that fetches real-time stock data, calculates technical indicators, and predicts next-day stock movements using **machine learning**.  


## Features

- **Real-time Stock Data**: Fetches historical and current stock prices from Yahoo Finance.
- **Technical Indicators**: Calculates SMA, EMA, RSI, MACD, Bollinger Bands, and more.
- **Machine Learning Predictions**: Uses Random Forest to predict the probability of next-day stock movement.
- **AI Strategy vs Buy & Hold**: Compare AI-driven trading strategy with traditional buy & hold.
- **Visualizations**: Interactive charts with Plotly for price trends, RSI, MACD, and feature importance.
- **Customizable Model**: Adjust Random Forest parameters and test size via sidebar.

## Technologies Used

- **Python 3.10**
- **Streamlit** (UI)
- **yfinance** (Stock data)
- **Pandas & NumPy** (Data processing)
- **Plotly** (Interactive visualizations)
- **Scikit-learn** (Machine learning)

## How to Run

1. Create a virtual environment:
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows

2: Install dependencies:
pip install -r requirements.txt

3: Run the Streamlit app:
streamlit run app.py
