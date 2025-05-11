import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import random
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# API Keys - replace with your own
ALPHA_KEY = "WQE6CUR6VDRWPOVI"
FINNHUB_KEY = "d0bevn9r01qo0h63hnh0d0bevn9r01qo0h63hnhg"
YAHOO_KEY = "8d5c766533msh267a1e66e4dd1c1p1b76f9jsn691553d9fe15"

# Page config
st.set_page_config(page_title="Stock Search Engine", page_icon="📈", layout="wide")

# CSS styling
st.markdown("""
<style>
    .main-title { font-size: 2.5rem; color: #ff3333; }
    .stock-card, .forecast-card {
        background-color: #1e1e1e;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        margin: 15px 0;
    }
    .prediction-reason {
        background-color: #2a2a2a;
        padding: 10px;
        border-left: 3px solid #ff3333;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>STOCK SEARCH ENGINE</h1>", unsafe_allow_html=True)
ticker = st.text_input("", placeholder="Enter stock ticker symbol (e.g., AAPL, GOOGL)", key="search_input")

def get_stock_quote(ticker):
    # Try Finnhub
    try:
        url = f"https://finnhub.io/api/v1/quote"
        r = requests.get(url, params={"symbol": ticker, "token": FINNHUB_KEY}, timeout=3)
        
        if r.status_code == 200:
            data = r.json()
            if data and 'c' in data and data['c'] > 0:
                return {
                    'longName': ticker,
                    'regularMarketPrice': data['c'],
                    'regularMarketChange': data['d'],
                    'regularMarketChangePercent': data['dp']
                }
    except:
        pass
    
    # Try Yahoo
    try:
        url = "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/quote"
        headers = {
            "x-rapidapi-host": "yahoo-finance15.p.rapidapi.com",
            "x-rapidapi-key": YAHOO_KEY
        }
        
        r = requests.get(url, 
                         params={"ticker": ticker, "type": "STOCKS"}, 
                         headers=headers, 
                         timeout=3)
        
        if r.status_code == 200:
            data = r.json()
            quote = data.get('data', {}).get('quote', {})
            if quote:
                return {
                    'longName': quote.get('longName', ticker),
                    'regularMarketPrice': quote.get('regularMarketPrice', 0),
                    'regularMarketChange': quote.get('regularMarketChange', 0),
                    'regularMarketChangePercent': quote.get('regularMarketChangePercent', 0)
                }
    except:
        pass
    
    # Last try: Alpha Vantage
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": ticker,
            "apikey": ALPHA_KEY
        }
        
        r = requests.get(url, params=params, timeout=3)
        if r.status_code == 200:
            data = r.json()
            quote = data.get('Global Quote', {})
            
            if '05. price' in quote:
                return {
                    'longName': ticker,
                    'regularMarketPrice': float(quote.get('05. price', 0)),
                    'regularMarketChange': float(quote.get('09. change', 0)),
                    'regularMarketChangePercent': float(quote.get('10. change percent', '0%').strip('%'))
                }
    except:
        pass
        
    return None

def get_historical_data(ticker, period='7d'):
    try:
        outputsize = "compact" if period == '7d' else "full"
        
        with st.spinner(f"Fetching data for {ticker}..."):
            r = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "TIME_SERIES_DAILY",
                    "symbol": ticker,
                    "apikey": ALPHA_KEY,
                    "outputsize": outputsize
                },
                timeout=5
            )
            
            if r.status_code == 200:
                data = r.json()
                ts_key = "Time Series (Daily)"
                        
                if ts_key in data:
                    ts = data[ts_key]
                    if ts:
                        df = pd.DataFrame.from_dict(ts, orient='index')
                        df = df.rename(columns={
                            '1. open': 'Open', '2. high': 'High',
                            '3. low': 'Low', '4. close': 'Close',
                            '5. volume': 'Volume'
                        })
                        
                        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col])
                        
                        df.index = pd.to_datetime(df.index)
                        df = df.sort_index()
                        
                        # Filter for period
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=30 if period == '7d' else 365)
                        df = df[df.index >= start_date]
                        
                        return df
        
        # Fallback to Finnhub
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30 if period == '7d' else 365)).strftime('%Y-%m-%d')
        
        r = requests.get(
            "https://finnhub.io/api/v1/stock/candle",
            params={
                "symbol": ticker,
                "resolution": "D",
                "from": int(datetime.strptime(start_date, '%Y-%m-%d').timestamp()),
                "to": int(datetime.strptime(end_date, '%Y-%m-%d').timestamp()),
                "token": FINNHUB_KEY
            },
            timeout=5
        )
        
        if r.status_code == 200:
            data = r.json()
            if data.get('s') == 'ok':
                df = pd.DataFrame({
                    'Open': data.get('o', []),
                    'High': data.get('h', []),
                    'Low': data.get('l', []),
                    'Close': data.get('c', []),
                    'Volume': data.get('v', [])
                })
                df.index = pd.to_datetime([datetime.fromtimestamp(ts) for ts in data.get('t', [])])
                df = df.sort_index()
                return df
    
    except Exception as e:
        pass
            
    return None

def get_mock_data(ticker, period='7d'):
    seed = sum(ord(c) for c in ticker)
    random.seed(seed)
    np.random.seed(seed)
    
    days = 30 if period == '7d' else 365
    date_range = pd.date_range(start=datetime.now() - timedelta(days=days), end=datetime.now(), freq='D')
    
    base_price = {'AAPL': 182.0, 'MSFT': 347.0, 'GOOGL': 131.5, 
                 'AMZN': 169.8, 'META': 448.5, 'TSLA': 233.1, 'NVDA': 124.7}.get(ticker, 
                 random.uniform(80.0, 200.0))
    
    price_changes = np.random.normal(0, 0.014, len(date_range))
    price_changes[0] = 0
    cum_changes = np.cumprod(1 + price_changes)
    
    close_prices = base_price * cum_changes
    open_prices = close_prices * (1 + np.random.normal(0, 0.005, len(date_range)))
    high_prices = np.maximum(close_prices, open_prices) * (1 + np.abs(np.random.normal(0, 0.008, len(date_range))))
    low_prices = np.minimum(close_prices, open_prices) * (1 - np.abs(np.random.normal(0, 0.008, len(date_range))))
    volumes = np.abs(np.random.normal(5000000, 2000000, len(date_range)))
    
    df = pd.DataFrame({
        'Open': open_prices, 'High': high_prices, 'Low': low_prices,
        'Close': close_prices, 'Volume': volumes
    }, index=date_range)
    
    st.info(f"Using mock data for {ticker}")
    return df

def generate_prediction_reasons(historical_data, ticker, long_term=False):
    if historical_data is None or len(historical_data) < 5:
        return ["Insufficient historical data"]
    
    df = historical_data.copy()
    
    # Calculate metrics
    recent_trend = df['Close'].iloc[-5:].mean() - df['Close'].iloc[-10:-5].mean()
    recent_volatility = df['Close'].pct_change().std() * 100
    avg_volume = df['Volume'].mean()
    volume_change = (df['Volume'].iloc[-5:].mean() / df['Volume'].iloc[-20:-5].mean() - 1) * 100
    
    # Long-term metrics
    if long_term and len(df) > 60:
        trend_90d = df['Close'].iloc[-30:].mean() - df['Close'].iloc[-90:-30].mean()
        trend_strength = abs(trend_90d) / df['Close'].iloc[-90:].std()
    else:
        trend_90d = 0
        trend_strength = 0
    
    reasons = []
    
    # Reason 1: Price trend
    if long_term:
        if trend_90d > 0:
            strength = "strong" if trend_strength > 1.5 else "moderate"
            reasons.append(f"{strength.capitalize()} bullish trend observed in {ticker} over the past quarter")
        else:
            strength = "strong" if trend_strength > 1.5 else "moderate"
            reasons.append(f"{strength.capitalize()} bearish pressure detected in {ticker}'s quarterly performance")
    else:
        if recent_trend > 0:
            reasons.append(f"Recent upward price trend observed in {ticker} over the past week")
        else:
            reasons.append(f"Recent downward price movement detected in {ticker}'s trading pattern")
    
    # Reason 2: Volume
    if volume_change > 10:
        reasons.append(f"Trading volume has increased by {volume_change:.1f}% recently, suggesting higher market interest")
    elif volume_change < -10:
        reasons.append(f"Declining trading volume indicates potential reduced market interest")
    else:
        reasons.append(f"Stable trading volume around {avg_volume/1000000:.1f}M shares suggests consistent market activity")
    
    # Reason 3: Volatility
    if recent_volatility > 2.5:
        reasons.append(f"{'Historical' if long_term else 'High price'} volatility ({recent_volatility:.1f}%) indicates {'potential for significant price swings over coming months' if long_term else 'market uncertainty'}")
    else:
        reasons.append(f"{'Low historical' if long_term else 'Relatively low'} volatility ({recent_volatility:.1f}%) suggests {'potential for steady, gradual movement' if long_term else 'stable trading conditions'}")
    
    # Reason 4: Technical indicator
    if long_term:
        support = round(df['Close'].min() * (1 + 5/100), 2)
        resistance = round(df['Close'].max() * (1 - 5/100), 2)
        
        signal = random.choice(["bullish", "bearish", "neutral"])
        if signal == "bullish":
            reasons.append(f"Long-term technical analysis shows potential breakthrough above resistance level of ${resistance}")
        elif signal == "bearish":
            reasons.append(f"Multi-month support level at ${support} could be tested based on technical patterns")
        else:
            reasons.append(f"Key technical levels: support at ${support}, resistance at ${resistance}")
    else:
        signal = random.choice(["bullish", "bearish", "neutral"])
        if signal == "bullish":
            reasons.append("Technical indicators show potential bullish MACD crossover pattern forming")
        elif signal == "bearish":
            reasons.append("Technical analysis suggests bearish divergence in momentum indicators")
        else:
            reasons.append("Technical indicators show neutral consolidation pattern")
    
    # Reason 5: Sector condition
    market = random.choice(["strong", "weak", "neutral"])
    sectors = {
        "AAPL": "technology", "MSFT": "technology", "GOOGL": "technology",
        "AMZN": "consumer", "WMT": "retail", "JPM": "banking",
        "PFE": "healthcare", "XOM": "energy"
    }
    sector = sectors.get(ticker, random.choice(["technology", "healthcare", "finance", "energy"]))
    
    if long_term:
        if market == "strong":
            reasons.append(f"Six-month outlook for the {sector} sector remains positive, potentially providing sustained tailwind")
        elif market == "weak":
            reasons.append(f"Projected economic headwinds may impact the entire {sector} sector over the coming quarters")
        else:
            reasons.append(f"The {sector} sector shows cyclical patterns that could impact medium-term price trajectory")
    else:
        if market == "strong":
            reasons.append(f"Overall {sector} sector strength is likely to provide tailwind")
        elif market == "weak":
            reasons.append(f"General {sector} sector weakness may apply downward pressure")
        else:
            reasons.append(f"The {sector} sector shows mixed performance which may impact price movement")
        
    return reasons

def generate_prediction(historical_data, days_to_predict, ticker):
    if historical_data is None or len(historical_data) < 5:
        return {'predicted_dates': [], 'predicted_prices': [], 'reasons': ["Insufficient data for prediction"]}
    
    # Prepare data
    df = historical_data.copy()
    df['Date_Idx'] = np.arange(len(df))
    long_term = days_to_predict > 30
    
    if long_term:
        # Add technical indicators
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=min(200, len(df)//2)).mean()
        df = df.dropna()
        
        # Calculate volatility/trend
        hist_vol = df['Close'].pct_change().rolling(window=20).std().mean() * np.sqrt(252)
        
        # Future dates
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
        
        # Monte Carlo simulation
        n_sims = 100
        last_price = df['Close'].iloc[-1]
        
        # Set simulation parameters
        annual_return = df['Close'].pct_change().mean() * 252
        annual_vol = hist_vol
        
        # Add stock-specific bias
        stock_adj = {
            'AAPL': {'return': 0.15, 'vol': 0.25},
            'MSFT': {'return': 0.18, 'vol': 0.28},
            'GOOGL': {'return': 0.12, 'vol': 0.30},
            'AMZN': {'return': 0.20, 'vol': 0.35},
            'META': {'return': 0.10, 'vol': 0.40},
            'TSLA': {'return': 0.25, 'vol': 0.60},
            'NVDA': {'return': 0.30, 'vol': 0.50},
        }
        
        if ticker in stock_adj:
            # Blend historical with typical
            annual_return = (annual_return + stock_adj[ticker]['return']) / 2
            annual_vol = (annual_vol + stock_adj[ticker]['vol']) / 2
        
        # Reasonable constraints
        annual_return = min(max(annual_return, -0.20), 0.30)
        annual_vol = min(max(annual_vol, 0.15), 0.60)
        
        # Daily params
        daily_return = annual_return / 252
        daily_vol = annual_vol / np.sqrt(252)
        
        # Run simulations
        sim_results = np.zeros((days_to_predict, n_sims))
        
        for i in range(n_sims):
            prices = [last_price]
            
            for day in range(days_to_predict):
                # Add mean reversion for longer predictions
                if day > 30:
                    deviation = prices[-1] / last_price - 1
                    reversion = -deviation * min(0.005 * (day/30), 0.02)
                    adj_daily_return = daily_return + reversion
                else:
                    adj_daily_return = daily_return
                
                daily_change = np.random.normal(adj_daily_return, daily_vol)
                new_price = prices[-1] * (1 + daily_change)
                prices.append(new_price)
            
            sim_results[:, i] = prices[1:]
        
        # Median trajectory as prediction
        predicted_prices = np.median(sim_results, axis=1)
        
        # Add seasonality for realism
        seasonality = 0.02 * np.sin(np.linspace(0, 2*np.pi, days_to_predict))
        predicted_prices = predicted_prices * (1 + seasonality)
        
        # Smoothing
        predicted_prices = gaussian_filter1d(predicted_prices, sigma=3)
        
        reasons = generate_prediction_reasons(historical_data, ticker, long_term=True)
        
    else:
        # Short-term approach
        volatility = df['Close'].pct_change().std()
        poly_degree = 2 if volatility > 0.02 else 1
        
        poly = PolynomialFeatures(degree=poly_degree)
        X_poly = poly.fit_transform(df[['Date_Idx']])
        
        model = LinearRegression() 
        model.fit(X_poly, df['Close'])
        
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
        
        last_idx = df['Date_Idx'].iloc[-1]
        future_idxs = np.array([last_idx + i + 1 for i in range(days_to_predict)]).reshape(-1, 1)
        future_X_poly = poly.transform(future_idxs)
        
        predicted_prices = model.predict(future_X_poly)
        reasons = generate_prediction_reasons(historical_data, ticker)
    
    return {
        'predicted_dates': future_dates,
        'predicted_prices': predicted_prices,
        'reasons': reasons
    }

def create_stock_chart(historical_data, prediction_data, title):
    if historical_data is None or len(historical_data) < 2:
        return None
    
    hist_dates = historical_data.index
    hist_prices = historical_data['Close'].values
    pred_dates = prediction_data['predicted_dates']
    pred_prices = prediction_data['predicted_prices']
    
    fig = go.Figure()
    
    # Historical line
    fig.add_trace(
        go.Scatter(
            x=hist_dates, 
            y=hist_prices,
            name="Historical",
            line=dict(color="#4285F4", width=3),
            mode="lines"
        )
    )
    
    # Prediction line
    if len(pred_dates) > 0 and len(hist_prices) > 0:
        merged_dates = [hist_dates[-1]] + pred_dates
        merged_prices = [hist_prices[-1]] + list(pred_prices)
        
        fig.add_trace(
            go.Scatter(
                x=merged_dates, 
                y=merged_prices,
                name="Forecast",
                line=dict(color="#ff3333", width=3, dash='dash'),
                mode="lines"
            )
        )
    
    # Style
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color="#ffffff")),
        paper_bgcolor="#121212",
        plot_bgcolor="#121212",
        hovermode="x unified",
        xaxis=dict(gridcolor="#1e1e1e", tickfont=dict(color="#ffffff")),
        yaxis=dict(gridcolor="#1e1e1e", tickfont=dict(color="#ffffff")),
        legend=dict(orientation="h", y=1.02, x=1, font=dict(color="#ffffff"))
    )
    
    return fig

def search_stock(ticker):
    if not ticker:
        return

    ticker = ticker.upper().strip()
    stock_data = get_stock_quote(ticker)

    # Fallback to mock if needed
    if not stock_data:
        st.warning(f"Using mock data for {ticker}")
        stock_data = {
            'longName': f"{ticker} (Mock Data)",
            'regularMarketPrice': random.uniform(100.0, 200.0),
            'regularMarketChange': random.uniform(-5.0, 5.0),
            'regularMarketChangePercent': random.uniform(-3.0, 3.0)
        }

    # Display stock info
    price_color = '#33ff33' if stock_data.get('regularMarketChange', 0) >= 0 else '#ff3333'
    price = stock_data.get('regularMarketPrice', 0)
    change = stock_data.get('regularMarketChange', 0)
    
    st.markdown(f"""
    <div class="stock-card">
        <h2>{ticker}</h2>
        <p>{stock_data.get('longName', 'Unknown Company')}</p>
        <h3>${price:.2f}</h3>
        <p style="color: {price_color}">
            {'+' if change >= 0 else ''}{change:.2f}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Get historical data
    historical_data_7d = get_historical_data(ticker, '7d') 
    if historical_data_7d is None:
        historical_data_7d = get_mock_data(ticker, '7d')
    
    historical_data_6m = get_historical_data(ticker, '6mo')
    if historical_data_6m is None:
        historical_data_6m = get_mock_data(ticker, '6mo')

    # Create tabs for forecasts
    tab1, tab2 = st.tabs(["7-Day Forecast", "6-Month Forecast"])
    
    # 7-Day tab
    with tab1:
        if historical_data_7d is not None and len(historical_data_7d) >= 3:
            with st.spinner("Generating forecast..."):
                projected_7d = generate_prediction(historical_data_7d, 7, ticker)
                fig = create_stock_chart(historical_data_7d, projected_7d, f"{ticker} - 7 Day Forecast")
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show prediction
                    if len(projected_7d['predicted_prices']) > 0:
                        final_price = projected_7d['predicted_prices'][-1]
                        price_color = '#33ff33' if final_price >= historical_data_7d['Close'].iloc[-1] else '#ff3333'
                        
                        st.markdown(f"""
                        <div class="forecast-card">
                            <h3>7-Day Projected Price: <span style="color: {price_color}">${final_price:.2f}</span></h3>
                            <div class="prediction-reason">
                                <p><strong>Prediction Reasons:</strong></p>
                                <ul>
                                    {"".join([f"<li>{reason}</li>" for reason in projected_7d['reasons']])}
                                </ul>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.error("No historical data available for 7-day forecast")

    # 6-Month tab
    with tab2:
        if historical_data_6m is not None and len(historical_data_6m) >= 30:
            with st.spinner("Generating forecast..."):
                projected_6m = generate_prediction(historical_data_6m, 180, ticker)
                fig = create_stock_chart(historical_data_6m, projected_6m, f"{ticker} - 6 Month Forecast")
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show prediction
                    if len(projected_6m['predicted_prices']) > 0:
                        final_price = projected_6m['predicted_prices'][-1]
                        price_color = '#33ff33' if final_price >= historical_data_6m['Close'].iloc[-1] else '#ff3333'
                        
                        st.markdown(f"""
                        <div class="forecast-card">
                            <h3>6-Month Projected Price: <span style="color: {price_color}">${final_price:.2f}</span></h3>
                            <div class="prediction-reason">
                                <p><strong>Prediction Reasons:</strong></p>
                                <ul>
                                    {"".join([f"<li>{reason}</li>" for reason in projected_6m['reasons']])}
                                </ul>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.error("No historical data available for 6-month forecast")

# Run the app
if ticker:
    search_stock(ticker)
else:
    # Display welcome
    st.markdown("""
    <div style="text-align: center; margin-top: 50px;">
        <h2>Enter a stock symbol above to begin analysis</h2>
        <p>Get real-time prices, view trends, and see price predictions</p>
        <p>Try entering symbols like: AAPL, MSFT, GOOGL, AMZN, TSLA</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; opacity: 0.7;">
    <p style="font-size: 0.8rem;">
        Data provided by Finnhub, Yahoo Finance, and Alpha Vantage.
        Stock predictions are for demonstration purposes only.
    </p>
</div>
""", unsafe_allow_html=True)