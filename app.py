import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Stock Market Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-container {
        background-color: #262730;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: #262730;
        padding: 1rem;
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🚀 AI Stock Market Analyzer Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
**Built with Python, Streamlit, and Machine Learning!** 🤖📊

This app pulls real-time stock data with yfinance
and predict the probability of next-day stock movement.
""")

# Sidebar for stock selection
st.sidebar.header("📊 Stock Selection")
default_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'SPY']
selected_stock = st.sidebar.selectbox("Select Stock Symbol", default_stocks + ['Custom'], index=0)

if selected_stock == 'Custom':
    selected_stock = st.sidebar.text_input("Enter Stock Symbol", "").upper()

# Date range selection
st.sidebar.header("📅 Date Range")
end_date = datetime.now()
start_date = st.sidebar.date_input("Start Date", end_date - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", end_date)

# Model parameters
st.sidebar.header("🎯 Model Parameters")
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100
n_estimators = st.sidebar.slider("Random Forest Trees", 50, 200, 100)

@st.cache_data
def fetch_stock_data(symbol, start, end):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(start=start, end=end)
        if data.empty:
            st.error(f"No data found for {symbol}")
            return None
        
        # Get additional info for display
        try:
            info = stock.info
            company_name = info.get('longName', symbol)
            currency = info.get('currency', 'USD')
            st.sidebar.success(f"✅ Loaded: {company_name}")
            st.sidebar.info(f"💰 Currency: {currency}")
        except:
            pass
            
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def calculate_sma(data, window):
    """Calculate Simple Moving Average"""
    return data.rolling(window=window).mean()

def calculate_ema(data, window):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=window).mean()

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = calculate_ema(macd_line, signal)
    macd_histogram = macd_line - macd_signal
    return macd_line, macd_signal, macd_histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(data, window)
    std = data.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def calculate_technical_indicators(data):
    """Calculate all technical indicators"""
    # Simple Moving Averages
    data['SMA_20'] = calculate_sma(data['Close'], 20)
    data['SMA_50'] = calculate_sma(data['Close'], 50)
    data['SMA_200'] = calculate_sma(data['Close'], 200)
    
    # Exponential Moving Averages
    data['EMA_12'] = calculate_ema(data['Close'], 12)
    data['EMA_26'] = calculate_ema(data['Close'], 26)
    
    # RSI
    data['RSI'] = calculate_rsi(data['Close'])
    
    # MACD
    data['MACD'], data['MACD_Signal'], data['MACD_Histogram'] = calculate_macd(data['Close'])
    
    # Bollinger Bands
    data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = calculate_bollinger_bands(data['Close'])
    
    # Volume indicators
    data['Volume_SMA'] = calculate_sma(data['Volume'], 20)
    
    # Price-based indicators
    data['Price_Change'] = data['Close'].pct_change()
    data['High_Low_Pct'] = (data['High'] - data['Low']) / data['Close']
    data['Close_Open_Pct'] = (data['Close'] - data['Open']) / data['Open']
    
    # Volatility
    data['Volatility'] = data['Price_Change'].rolling(window=20).std()
    
    # Target variable
    data['Next_Day_Up'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    return data

def prepare_features(data):
    """Prepare features for machine learning"""
    # Technical indicators as features
    feature_columns = [
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
        'Price_Change', 'High_Low_Pct', 'Close_Open_Pct',
        'Volatility'
    ]
    
    # Add moving average ratios
    data['SMA_Ratio_20_50'] = data['SMA_20'] / data['SMA_50']
    data['SMA_Ratio_50_200'] = data['SMA_50'] / data['SMA_200']
    data['Price_to_SMA20'] = data['Close'] / data['SMA_20']
    data['Price_to_SMA50'] = data['Close'] / data['SMA_50']
    
    feature_columns.extend(['SMA_Ratio_20_50', 'SMA_Ratio_50_200', 'Price_to_SMA20', 'Price_to_SMA50'])
    
    # Volume indicators
    data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
    data['Volume_Price_Trend'] = (data['Volume'] * data['Price_Change'])
    feature_columns.extend(['Volume_Ratio', 'Volume_Price_Trend'])
    
    # Bollinger Bands position
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
    feature_columns.append('BB_Position')
    
    # RSI levels
    data['RSI_Oversold'] = (data['RSI'] < 30).astype(int)
    data['RSI_Overbought'] = (data['RSI'] > 70).astype(int)
    feature_columns.extend(['RSI_Oversold', 'RSI_Overbought'])
    
    # Create feature matrix
    features_df = data[feature_columns].dropna()
    target = data['Next_Day_Up'].dropna()
    
    # Align features and target
    min_length = min(len(features_df), len(target))
    features_df = features_df.iloc[-min_length:]
    target = target.iloc[-min_length:]
    
    return features_df, target

def train_model(X, y, test_size, n_estimators):
    """Train Random Forest model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=42, 
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
    }
    
    return model, metrics, X_test, y_test, y_pred_proba

def create_prediction_chart(data, probabilities):
    """Create prediction probability chart"""
    fig = go.Figure()
    
    # Get last N days for visualization
    n_days = min(100, len(probabilities))
    dates = data.index[-n_days:]
    probs = probabilities[-n_days:]
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=probs,
        mode='lines',
        name='Prediction Probability',
        line=dict(color='#00D4FF', width=2),
        fill='tonexty',
        fillcolor='rgba(0, 212, 255, 0.1)'
    ))
    
    # Add threshold line
    fig.add_hline(y=0.6, line_dash="dash", line_color="red", 
                  annotation_text="Buy Threshold (60%)")
    fig.add_hline(y=0.4, line_dash="dash", line_color="green", 
                  annotation_text="Sell Threshold (40%)")
    
    fig.update_layout(
        title="Predicted Probability of Next-Day UP",
        xaxis_title="Date",
        yaxis_title="Probability",
        template="plotly_dark",
        height=400,
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def calculate_strategy_performance(data, predictions, probabilities, threshold=0.6):
    """Calculate AI strategy vs Buy & Hold performance"""
    # AI Strategy: Buy when probability > threshold
    data = data.copy()
    data['AI_Signal'] = (probabilities > threshold).astype(int)
    data['AI_Returns'] = data['AI_Signal'].shift(1) * data['Price_Change']
    data['Buy_Hold_Returns'] = data['Price_Change']
    
    # Calculate cumulative returns
    data['AI_Cumulative'] = (1 + data['AI_Returns'].fillna(0)).cumprod()
    data['BuyHold_Cumulative'] = (1 + data['Buy_Hold_Returns'].fillna(0)).cumprod()
    
    # Performance metrics
    ai_total_return = (data['AI_Cumulative'].iloc[-1] - 1) * 100
    bh_total_return = (data['BuyHold_Cumulative'].iloc[-1] - 1) * 100
    
    # Calculate additional metrics
    ai_volatility = data['AI_Returns'].std() * np.sqrt(252) * 100
    bh_volatility = data['Buy_Hold_Returns'].std() * np.sqrt(252) * 100
    
    # Sharpe ratio (assuming risk-free rate of 2%)
    ai_sharpe = (ai_total_return - 2) / ai_volatility if ai_volatility > 0 else 0
    bh_sharpe = (bh_total_return - 2) / bh_volatility if bh_volatility > 0 else 0
    
    # Max drawdown
    ai_running_max = data['AI_Cumulative'].expanding().max()
    ai_drawdown = ((data['AI_Cumulative'] - ai_running_max) / ai_running_max * 100).min()
    
    bh_running_max = data['BuyHold_Cumulative'].expanding().max()
    bh_drawdown = ((data['BuyHold_Cumulative'] - bh_running_max) / bh_running_max * 100).min()
    
    performance_metrics = {
        'ai_return': ai_total_return,
        'bh_return': bh_total_return,
        'ai_volatility': ai_volatility,
        'bh_volatility': bh_volatility,
        'ai_sharpe': ai_sharpe,
        'bh_sharpe': bh_sharpe,
        'ai_drawdown': ai_drawdown,
        'bh_drawdown': bh_drawdown
    }
    
    return data, performance_metrics

def create_price_chart(data):
    """Create price chart with technical indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Price & Moving Averages', 'RSI', 'MACD'),
        vertical_spacing=0.08,
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Price and moving averages
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], 
                            name='Close Price', line=dict(color='white')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], 
                            name='SMA 20', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], 
                            name='SMA 50', line=dict(color='blue')), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], 
                            name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], 
                            name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], 
                            name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], 
                            name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], 
                            name='Signal', line=dict(color='red')), row=3, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data['MACD_Histogram'], 
                        name='Histogram', marker_color='green'), row=3, col=1)
    
    fig.update_layout(height=800, template="plotly_dark", showlegend=True)
    return fig

# Main application
if st.sidebar.button("🚀 Run Analysis", type="primary"):
    with st.spinner(f"Analyzing {selected_stock}..."):
        # Fetch data
        stock_data = fetch_stock_data(selected_stock, start_date, end_date)
        
        if stock_data is not None:
            # Calculate indicators
            stock_data = calculate_technical_indicators(stock_data)
            
            # Prepare features
            X, y = prepare_features(stock_data)
            
            if len(X) > 50:  # Ensure enough data
                # Train model
                model, metrics, X_test, y_test, probabilities = train_model(X, y, test_size, n_estimators)
                
                # Model Performance Metrics
                st.markdown("## 📊 Model Performance")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Accuracy",
                        f"{metrics['accuracy']:.2%}",
                        delta=f"{(metrics['accuracy'] - 0.5):.2%}"
                    )
                
                with col2:
                    st.metric(
                        "Precision", 
                        f"{metrics['precision']:.2%}",
                        delta=f"{(metrics['precision'] - 0.5):.2%}"
                    )
                
                with col3:
                    st.metric(
                        "Recall",
                        f"{metrics['recall']:.2%}",
                        delta=f"{(metrics['recall'] - 0.5):.2%}"
                    )
                
                with col4:
                    st.metric(
                        "ROC-AUC",
                        f"{metrics['roc_auc']:.3f}",
                        delta=f"{(metrics['roc_auc'] - 0.5):.3f}"
                    )
                
                # Technical Analysis Chart
                st.markdown("## 📈 Technical Analysis")
                tech_chart = create_price_chart(stock_data.tail(100))
                st.plotly_chart(tech_chart, use_container_width=True)
                
                # Prediction Chart
                st.markdown("## 🎯 Predicted Probability of Next-Day UP")
                prediction_chart = create_prediction_chart(stock_data, probabilities)
                st.plotly_chart(prediction_chart, use_container_width=True)
                
                # Recent Predictions Table
                st.markdown("## 📋 Recent Predictions")
                recent_data = stock_data.tail(10).copy()
                recent_probs = probabilities[-10:] if len(probabilities) >= 10 else probabilities
                
                recent_predictions = pd.DataFrame({
                    'Date': recent_data.index.strftime('%Y-%m-%d'),
                    'Close Price': recent_data['Close'].round(2),
                    'RSI': recent_data['RSI'].round(2),
                    'MACD': recent_data['MACD'].round(4),
                    'Volume': recent_data['Volume'].astype(int),
                    'Prediction Probability': [f"{p:.2%}" for p in recent_probs],
                    'Signal': ['BUY' if p > 0.6 else 'SELL' for p in recent_probs]
                })
                
                st.dataframe(recent_predictions, use_container_width=True)
                
                # AI Strategy vs Buy & Hold
                st.markdown("## 💰 AI Strategy vs Buy & Hold")
                
                strategy_data, perf_metrics = calculate_strategy_performance(
                    stock_data[-len(probabilities):], None, probabilities
                )
                
                # Performance metrics display
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "AI Strategy Return",
                        f"{perf_metrics['ai_return']:.2f}%",
                        delta=f"{perf_metrics['ai_return'] - perf_metrics['bh_return']:.2f}%"
                    )
                
                with col2:
                    st.metric(
                        "Buy & Hold Return",
                        f"{perf_metrics['bh_return']:.2f}%"
                    )
                
                with col3:
                    st.metric(
                        "AI Sharpe Ratio",
                        f"{perf_metrics['ai_sharpe']:.2f}",
                        delta=f"{perf_metrics['ai_sharpe'] - perf_metrics['bh_sharpe']:.2f}"
                    )
                
                with col4:
                    st.metric(
                        "AI Max Drawdown",
                        f"{perf_metrics['ai_drawdown']:.2f}%",
                        delta=f"{perf_metrics['ai_drawdown'] - perf_metrics['bh_drawdown']:.2f}%"
                    )
                
                # Strategy Performance Chart
                fig_strategy = go.Figure()
                
                fig_strategy.add_trace(go.Scatter(
                    x=strategy_data.index,
                    y=(strategy_data['AI_Cumulative'] - 1) * 100,
                    mode='lines',
                    name='AI Strategy',
                    line=dict(color='#00D4FF', width=2)
                ))
                
                fig_strategy.add_trace(go.Scatter(
                    x=strategy_data.index,
                    y=(strategy_data['BuyHold_Cumulative'] - 1) * 100,
                    mode='lines',
                    name='Buy & Hold',
                    line=dict(color='#FF6B6B', width=2)
                ))
                
                fig_strategy.update_layout(
                    title="Cumulative Returns: AI Strategy vs Buy & Hold",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return (%)",
                    template="plotly_dark",
                    height=400,
                    legend=dict(x=0, y=1)
                )
                
                st.plotly_chart(fig_strategy, use_container_width=True)
                
                # Feature Importance
                st.markdown("## 🎯 Feature Importance")
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                fig_importance = px.bar(
                    feature_importance, 
                    x='Importance', 
                    y='Feature', 
                    orientation='h',
                    template='plotly_dark',
                    title='Top 10 Most Important Features'
                )
                fig_importance.update_layout(height=400)
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Final AI Suggestion
                latest_prob = probabilities[-1] if len(probabilities) > 0 else 0.5
                
                st.markdown("## 🎯 Final AI Suggestion")
                
                if latest_prob > 0.6:
                    st.success(f"🚀 **BUY SIGNAL** - Probability: {latest_prob:.2%}")
                    st.markdown("The model predicts a high probability of upward movement tomorrow.")
                elif latest_prob < 0.4:
                    st.error(f"📉 **SELL SIGNAL** - Probability: {latest_prob:.2%}")
                    st.markdown("The model predicts a low probability of upward movement tomorrow.")
                else:
                    st.warning(f"⚠️ **NEUTRAL** - Probability: {latest_prob:.2%}")
                    st.markdown("The model shows uncertainty. Consider waiting for a clearer signal.")
                
                st.markdown("---")
                st.markdown("*#AI #MachineLearning #Finance #Python #Streamlit #StockMarket #DataScience*")
                
            else:
                st.error("Not enough data to train the model. Please select a longer date range.")
        else:
            st.error("Unable to fetch stock data. Please check the stock symbol and try again.")

else:
    st.info("👆 Select a stock symbol and click 'Run Analysis' to start!")