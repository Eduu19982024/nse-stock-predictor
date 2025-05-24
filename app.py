import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from xgboost import XGBClassifier, XGBRegressor
import streamlit as st
import snscrape.modules.twitter as sntwitter
from transformers import pipeline

nse_symbols = [
    # Nifty 50 stocks
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "KOTAKBANK.NS", "ITC.NS", "LT.NS", "AXISBANK.NS",
    "HINDUNILVR.NS", "BAJFINANCE.NS", "MARUTI.NS", "NESTLEIND.NS", "HDFC.NS",
    "ASIANPAINT.NS", "SUNPHARMA.NS", "BHARTIARTL.NS", "WIPRO.NS", "ULTRACEMCO.NS",
    "TECHM.NS", "TITAN.NS", "ONGC.NS", "POWERGRID.NS", "DRREDDY.NS",
    "JSWSTEEL.NS", "COALINDIA.NS", "BAJAJ-AUTO.NS", "GRASIM.NS", "HCLTECH.NS",
    "EICHERMOT.NS", "DIVISLAB.NS", "ADANIPORTS.NS", "TATASTEEL.NS", "BPCL.NS",
    "CIPLA.NS", "M&M.NS", "HDFCLIFE.NS", "SBILIFE.NS", "NTPC.NS",
    "IOC.NS", "UPL.NS", "HEROMOTOCO.NS", "BAJAJFINSV.NS", "SHREECEM.NS",

    # Bank Nifty stocks (major banks in Bank Nifty index)
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "INDUSINDBK.NS", "PNB.NS", "IDFCFIRSTB.NS", "BANKBARODA.NS", "FEDERALBNK.NS",
    "RBLBANK.NS", "YESBANK.NS", "IDBI.NS", "BANDHANBNK.NS", "AUROPHARMA.NS",

    # Additional large/mid-cap stocks popular in NSE
    "VEDL.NS", "GAIL.NS", "TATAMOTORS.NS", "HINDALCO.NS", "DLF.NS",
    "LICHSGFIN.NS", "HAVELLS.NS", "TATAELXSI.NS", "PIDILITIND.NS", "ZEEL.NS",
    "CROMPTON.NS", "SBICARD.NS", "ADANIGREEN.NS", "AMBUJACEM.NS", "EXIDEIND.NS",
    "MOTHERSUMI.NS", "PETRONET.NS", "M&MFIN.NS", "BIOCON.NS", "ABB.NS",
    "DABUR.NS", "ICICIGI.NS", "CADILAHC.NS", "ICICIPRULI.NS", "COLPAL.NS",
    "INDIGO.NS", "JINDALSTEL.NS", "SRF.NS", "HDFCAMC.NS", "NMDC.NS",
    "SHRIRAMFIN.NS", "BHEL.NS", "PAGEIND.NS", "BOSCHLTD.NS", "LTI.NS",
    "TATACONSUM.NS", "HINDPETRO.NS", "SRTRANSFIN.NS", "IDFC.NS", "JSWENERGY.NS",
    "ASHOKLEY.NS", "HINDZINC.NS", "TORNTPOWER.NS"
]


def get_stock_data(ticker):
    try:
        df = yf.download(ticker, period='6mo', interval='1d', progress=False)
        return df.dropna()
    except:
        return pd.DataFrame()

def add_features(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['ema'] = ta.trend.EMAIndicator(df['Close'], window=10).ema_indicator()
    df['macd'] = ta.trend.MACD(df['Close']).macd_diff()
    df['sma10'] = df['Close'].rolling(10).mean()
    df['sma50'] = df['Close'].rolling(50).mean()
    df['bbp'] = (df['Close'] - df['Close'].rolling(20).mean()) / (2 * df['Close'].rolling(20).std())
    df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df.dropna()

def get_sentiment_score(stock_name):
    try:
        tweets = sntwitter.TwitterSearchScraper(f'{stock_name} stock since:2025-05-18').get_items()
        text_data = [tweet.content for tweet in list(tweets)[:30]]
        sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        results = sentiment_pipeline(text_data)
        scores = [1 if r['label'] == 'positive' else -1 if r['label'] == 'negative' else 0 for r in results]
        return sum(scores) / len(scores) if scores else 0
    except:
        return 0

def prepare_targets(df, days):
    df['target_dir'] = (df['Close'].shift(-days) > df['Close']).astype(int)
    df['target_pct'] = (df['Close'].shift(-days) - df['Close']) / df['Close'] * 100
    return df.dropna()

def train_models(df):
    features = ['rsi', 'ema', 'macd', 'sma10', 'sma50', 'bbp', 'vwap', 'sentiment']
    X = df[features]
    y_class = df['target_dir']
    y_reg = df['target_pct']

    X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, shuffle=False)
    _, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, shuffle=False)

    model_class = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model_reg = XGBRegressor()
    model_class.fit(X_train, y_class_train)
    model_reg.fit(X_train, y_reg_train)

    y_pred_class = model_class.predict(X_test)
    y_pred_reg = model_reg.predict(X_test)

    acc = accuracy_score(y_class_test, y_pred_class)
    mae = mean_absolute_error(y_reg_test, y_pred_reg)

    last_input = df[features].iloc[-1:].values
    dir_prob = model_class.predict_proba(last_input)[0][1]
    direction = "↑" if dir_prob >= 0.5 else "↓"
    pct_pred = model_reg.predict(last_input)[0]

    return direction, pct_pred, dir_prob, acc, mae

def main():
    st.title("📈 NSE Stock Forecast (Direction + % Movement)")
    days = st.slider("Days Ahead to Predict", 1, 10, 5)

    results = []

    for symbol in nse_symbols:
        df = get_stock_data(symbol)
        if df.empty: continue
        df = add_features(df)
        sentiment = get_sentiment_score(symbol.split('.')[0])
        df['sentiment'] = sentiment
        df = prepare_targets(df, days)
        if df.empty: continue

        try:
            direction, pct_pred, prob, acc, mae = train_models(df)
            results.append({
                "Stock": symbol,
                "Direction": direction,
                "Predicted %": round(pct_pred, 2),
                "Confidence": f"{round(prob*100, 1)}%",
                "Sentiment": round(sentiment, 2),
                "Accuracy": round(acc*100, 2),
                "MAE": round(mae, 2)
            })
        except:
            continue

    st.dataframe(pd.DataFrame(results).sort_values(by='Predicted %', ascending=False))

if __name__ == '__main__':
    main()
