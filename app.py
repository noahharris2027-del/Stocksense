from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

app = Flask(__name__)

# ──────────────────────────────────────────────
# Technical Indicators
# ──────────────────────────────────────────────

def sma(s, w): return s.rolling(w).mean()
def ema(s, sp): return s.ewm(span=sp, adjust=False).mean()

def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0).rolling(p).mean()
    l = (-d.where(d < 0, 0.0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l))

def macd(s, f=12, sl=26, sig=9):
    ml = ema(s, f) - ema(s, sl)
    si = ema(ml, sig)
    return ml, si, ml - si

def bollinger(s, w=20, n=2):
    m = sma(s, w); sd = s.rolling(w).std()
    return m + sd*n, m, m - sd*n

def atr(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def stochastic(h, l, c, kp=14, dp=3):
    ll = l.rolling(kp).min(); hh = h.rolling(kp).max()
    k = 100 * (c - ll) / (hh - ll)
    return k, k.rolling(dp).mean()

def obv(c, v):
    o = [0]
    for i in range(1, len(c)):
        o.append(o[-1] + (v.iloc[i] if c.iloc[i] > c.iloc[i-1] else -v.iloc[i] if c.iloc[i] < c.iloc[i-1] else 0))
    return pd.Series(o, index=c.index)

def adx_calc(h, l, c, p=14):
    pdm = h.diff(); mdm = -l.diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
    mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
    a = atr(h, l, c, p)
    pdi = 100 * ema(pdm, p) / a
    mdi = 100 * ema(mdm, p) / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return ema(dx, p), pdi, mdi

def vwap(h, l, c, v):
    tp = (h + l + c) / 3
    return (tp * v).cumsum() / v.cumsum()

def calc_fibonacci(high_val, low_val):
    diff = high_val - low_val
    levels = {
        '0%': high_val,
        '23.6%': high_val - 0.236 * diff,
        '38.2%': high_val - 0.382 * diff,
        '50%': high_val - 0.5 * diff,
        '61.8%': high_val - 0.618 * diff,
        '78.6%': high_val - 0.786 * diff,
        '100%': low_val,
    }
    return {k: round(v, 2) for k, v in levels.items()}


# ──────────────────────────────────────────────
# Signal Engine
# ──────────────────────────────────────────────

def generate_signals(df):
    signals = []
    score = 0
    c = df['Close'].iloc[-1]
    s50 = df['SMA_50'].iloc[-1]; s200 = df['SMA_200'].iloc[-1]
    r = df['RSI'].iloc[-1]
    m = df['MACD'].iloc[-1]; ms = df['MACD_Signal'].iloc[-1]
    pm = df['MACD'].iloc[-2]; ps = df['MACD_Signal'].iloc[-2]
    bbu = df['BB_Upper'].iloc[-1]; bbl = df['BB_Lower'].iloc[-1]
    sk = df['Stoch_K'].iloc[-1]; sd = df['Stoch_D'].iloc[-1]
    ax = df['ADX'].iloc[-1]; pdi = df['Plus_DI'].iloc[-1]; mdi = df['Minus_DI'].iloc[-1]

    # Golden/Death Cross
    if s50 > s200:
        signals.append(('bullish','Golden Cross','SMA 50 above SMA 200 — long-term uptrend')); score += 15
    else:
        signals.append(('bearish','Death Cross','SMA 50 below SMA 200 — long-term downtrend')); score -= 15

    # Price vs SMA
    if c > s50:
        signals.append(('bullish','Above SMA 50',f'Price ${c:.2f} > SMA50 ${s50:.2f}')); score += 10
    else:
        signals.append(('bearish','Below SMA 50',f'Price ${c:.2f} < SMA50 ${s50:.2f}')); score -= 10

    if c > s200:
        signals.append(('bullish','Above SMA 200',f'Price above 200-day moving average')); score += 8
    else:
        signals.append(('bearish','Below SMA 200',f'Price below 200-day moving average')); score -= 8

    # RSI
    if r < 30:
        signals.append(('bullish','RSI Oversold',f'RSI {r:.1f} — potential bounce')); score += 12
    elif r > 70:
        signals.append(('bearish','RSI Overbought',f'RSI {r:.1f} — may be overextended')); score -= 12
    elif r > 50:
        signals.append(('bullish','RSI Bullish',f'RSI {r:.1f} — momentum favors bulls')); score += 5
    else:
        signals.append(('bearish','RSI Bearish',f'RSI {r:.1f} — momentum favors bears')); score -= 5

    # MACD
    if m > ms:
        signals.append(('bullish','MACD Bullish','MACD above signal line')); score += 10
        if pm <= ps: signals.append(('bullish','MACD Crossover','Fresh bullish crossover')); score += 8
    else:
        signals.append(('bearish','MACD Bearish','MACD below signal line')); score -= 10
        if pm >= ps: signals.append(('bearish','MACD Crossunder','Fresh bearish crossunder')); score -= 8

    # Bollinger
    bb_pct = (c - bbl) / (bbu - bbl) if (bbu - bbl) != 0 else 0.5
    if bb_pct < 0.1:
        signals.append(('bullish','Lower Bollinger Band','Near lower band — potential support')); score += 8
    elif bb_pct > 0.9:
        signals.append(('bearish','Upper Bollinger Band','Near upper band — potential resistance')); score -= 8

    # Stochastic
    if sk < 20 and sd < 20:
        signals.append(('bullish','Stochastic Oversold',f'%K={sk:.1f} %D={sd:.1f}')); score += 8
    elif sk > 80 and sd > 80:
        signals.append(('bearish','Stochastic Overbought',f'%K={sk:.1f} %D={sd:.1f}')); score -= 8

    # ADX
    if ax > 25:
        td = 'bullish' if pdi > mdi else 'bearish'
        signals.append((td, f'Strong Trend ADX={ax:.1f}',f'+DI={pdi:.1f} -DI={mdi:.1f}'))
        score += 5 if td == 'bullish' else -5
    else:
        signals.append(('neutral',f'Weak Trend ADX={ax:.1f}','No strong directional trend'))

    # Volume
    vsma = df['Volume'].rolling(20).mean().iloc[-1]
    cv = df['Volume'].iloc[-1]
    if cv > vsma * 1.5:
        d = 'bullish' if c > df['Close'].iloc[-2] else 'bearish'
        signals.append((d,'High Volume',f'{cv/vsma:.1f}x 20-day avg'))
        score += 5 if d == 'bullish' else -5

    return signals, max(-100, min(100, score))


def predict_trend(df, score):
    c = df['Close'].iloc[-1]
    a = df['ATR'].iloc[-1]
    r5 = (c / df['Close'].iloc[-6] - 1) * 100 if len(df) > 5 else 0
    r20 = (c / df['Close'].iloc[-21] - 1) * 100 if len(df) > 20 else 0
    r60 = (c / df['Close'].iloc[-61] - 1) * 100 if len(df) > 60 else 0
    dv = df['Close'].pct_change().std() * 100
    av = dv * np.sqrt(252)

    if score > 30: outlook, color = 'Strongly Bullish', '#00c853'
    elif score > 10: outlook, color = 'Moderately Bullish', '#69f0ae'
    elif score > -10: outlook, color = 'Neutral / Sideways', '#ffd54f'
    elif score > -30: outlook, color = 'Moderately Bearish', '#ff8a65'
    else: outlook, color = 'Strongly Bearish', '#ff1744'

    high60 = df['High'].tail(60).max()
    low60 = df['Low'].tail(60).min()
    fib = calc_fibonacci(high60, low60)

    return {
        'outlook': outlook, 'color': color, 'score': score,
        'price': round(c, 2),
        'support': round(low60, 2), 'resistance': round(high60, 2),
        'bull_target': round(c + a * 3, 2), 'bear_target': round(c - a * 3, 2),
        'atr': round(a, 2),
        'r5': round(r5, 2), 'r20': round(r20, 2), 'r60': round(r60, 2),
        'daily_vol': round(dv, 3), 'annual_vol': round(av, 2),
        'fibonacci': fib,
        'high_52w': round(df['High'].tail(252).max(), 2) if len(df) >= 252 else round(df['High'].max(), 2),
        'low_52w': round(df['Low'].tail(252).min(), 2) if len(df) >= 252 else round(df['Low'].min(), 2),
    }


# ──────────────────────────────────────────────
# Broker Data (comprehensive)
# ──────────────────────────────────────────────

BROKERS = [
    {"name":"Fidelity","type":"Full Service","commission":"$0 stocks/ETFs","options":"$0.65/contract","min":"$0","fractional":True,"crypto":False,"rating":4.8,
     "pros":"Excellent research, no account fees, strong retirement tools, zero-fee index funds","cons":"Platform can feel complex for beginners","best_for":"Long-term investors, retirement accounts","url":"fidelity.com"},
    {"name":"Charles Schwab","type":"Full Service","commission":"$0 stocks/ETFs","options":"$0.65/contract","min":"$0","fractional":True,"crypto":True,"rating":4.7,
     "pros":"Great research & education, thinkorswim platform (from TD merge), excellent customer service","cons":"Crypto limited to select coins","best_for":"All-around investing, education-focused beginners","url":"schwab.com"},
    {"name":"Vanguard","type":"Full Service","commission":"$0 stocks/ETFs","options":"$1/contract","min":"$0","fractional":True,"crypto":False,"rating":4.5,
     "pros":"Lowest-cost index funds in industry, ideal for buy-and-hold, excellent retirement accounts","cons":"Outdated interface, no crypto, limited trading tools","best_for":"Passive index investors, retirement savers","url":"vanguard.com"},
    {"name":"Robinhood","type":"App-First","commission":"$0 stocks/ETFs","options":"$0/contract","min":"$0","fractional":True,"crypto":True,"rating":4.2,
     "pros":"Sleek mobile UI, easy to start, free options & crypto, IRA match 1-3%","cons":"Limited research, gamified design, fewer investment types","best_for":"Beginners wanting simplicity, mobile-first users","url":"robinhood.com"},
    {"name":"E*TRADE (Morgan Stanley)","type":"Full Service","commission":"$0 stocks/ETFs","options":"$0.65/contract","min":"$0","fractional":False,"crypto":False,"rating":4.4,
     "pros":"Power E*TRADE platform, strong options tools, good education","cons":"No fractional shares, no crypto","best_for":"Options traders, intermediate investors","url":"etrade.com"},
    {"name":"Interactive Brokers","type":"Advanced","commission":"$0 (Lite)","options":"$0.65/contract","min":"$0","fractional":True,"crypto":True,"rating":4.6,
     "pros":"Lowest margin rates, global market access (150+ markets), professional tools","cons":"Complex platform, not beginner-friendly","best_for":"Active traders, international investors, professionals","url":"interactivebrokers.com"},
    {"name":"Webull","type":"App-First","commission":"$0 stocks/ETFs","options":"$0/contract","min":"$0","fractional":True,"crypto":True,"rating":4.3,
     "pros":"Advanced charting on mobile, free Level 2 data, extended hours trading","cons":"Limited research, no mutual funds","best_for":"Active mobile traders, chart enthusiasts","url":"webull.com"},
    {"name":"SoFi Invest","type":"App-First","commission":"$0 stocks/ETFs","options":"N/A","min":"$1","fractional":True,"crypto":True,"rating":4.0,
     "pros":"All-in-one finance app, automated investing, crypto, IPO access","cons":"No options trading, limited tools","best_for":"Beginners wanting banking + investing in one app","url":"sofi.com"},
    {"name":"Merrill Edge","type":"Full Service","commission":"$0 stocks/ETFs","options":"$0.65/contract","min":"$0","fractional":True,"crypto":False,"rating":4.4,
     "pros":"Preferred Rewards with BofA (bonus rates), strong research (BofA/Merrill reports)","cons":"No crypto, fewer tools than competitors","best_for":"Bank of America customers, rewards-focused investors","url":"merrilledge.com"},
    {"name":"TD Ameritrade (→ Schwab)","type":"Full Service","commission":"$0 stocks/ETFs","options":"$0.65/contract","min":"$0","fractional":False,"crypto":False,"rating":4.6,
     "pros":"thinkorswim is top-tier, excellent education, migrating to Schwab","cons":"Being absorbed into Schwab, legacy accounts transitioning","best_for":"Active traders (thinkorswim users)","url":"tdameritrade.com"},
    {"name":"Tastytrade","type":"Specialized","commission":"$0 stocks","options":"$1/contract (capped)","min":"$0","fractional":False,"crypto":True,"rating":4.3,
     "pros":"Built for options traders, capped commissions, great options analytics","cons":"Limited stock research, niche platform","best_for":"Options-focused traders","url":"tastytrade.com"},
    {"name":"Ally Invest","type":"Full Service","commission":"$0 stocks/ETFs","options":"$0.50/contract","min":"$0","fractional":False,"crypto":True,"rating":4.1,
     "pros":"Low options fees, integrated with Ally Bank, robo-advisor option","cons":"Basic research tools, no fractional shares","best_for":"Ally Bank customers, budget-conscious investors","url":"ally.com/invest"},
    {"name":"Public","type":"App-First","commission":"$0 stocks/ETFs","options":"$0/contract","min":"$1","fractional":True,"crypto":True,"rating":4.1,
     "pros":"Social features, Treasury bills, alternative investments, no PFOF","cons":"Limited tools, newer platform","best_for":"Social investors, alternative asset curious","url":"public.com"},
    {"name":"Firstrade","type":"Discount","commission":"$0 stocks/ETFs","options":"$0/contract","min":"$0","fractional":False,"crypto":False,"rating":3.9,
     "pros":"Completely free trading including options, no minimums","cons":"Basic platform, limited research, customer service","best_for":"Cost-conscious traders who want free options","url":"firstrade.com"},
    {"name":"J.P. Morgan Self-Directed","type":"Full Service","commission":"$0 stocks/ETFs","options":"$0.65/contract","min":"$0","fractional":True,"crypto":False,"rating":4.2,
     "pros":"Chase integration, clean UI, portfolio builder tool","cons":"Limited research vs Fidelity/Schwab, no crypto","best_for":"Chase banking customers","url":"chase.com/personal/investments"},
    {"name":"moomoo","type":"App-First","commission":"$0 stocks/ETFs","options":"$0/contract","min":"$0","fractional":True,"crypto":True,"rating":4.2,
     "pros":"Free Level 2 data, advanced charting, paper trading, 24/5 trading","cons":"Newer to US market, limited track record","best_for":"Data-hungry traders wanting free advanced tools","url":"moomoo.com"},
]


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze')
def analyze():
    ticker = request.args.get('ticker', '^GSPC')
    period = request.args.get('period', '2y')
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return jsonify({'error': 'No data found'}), 404

        info = {}
        try: info = stock.info or {}
        except: pass

        # Indicators
        df['SMA_20'] = sma(df['Close'], 20)
        df['SMA_50'] = sma(df['Close'], 50)
        df['SMA_200'] = sma(df['Close'], 200)
        df['EMA_12'] = ema(df['Close'], 12)
        df['EMA_26'] = ema(df['Close'], 26)
        df['RSI'] = rsi(df['Close'])
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = macd(df['Close'])
        df['BB_Upper'], df['BB_Mid'], df['BB_Lower'] = bollinger(df['Close'])
        df['ATR'] = atr(df['High'], df['Low'], df['Close'])
        df['Stoch_K'], df['Stoch_D'] = stochastic(df['High'], df['Low'], df['Close'])
        df['OBV'] = obv(df['Close'], df['Volume'])
        df['ADX'], df['Plus_DI'], df['Minus_DI'] = adx_calc(df['High'], df['Low'], df['Close'])
        df['VWAP'] = vwap(df['High'], df['Low'], df['Close'], df['Volume'])

        dc = df.dropna()
        if len(dc) < 5:
            return jsonify({'error': 'Insufficient data'}), 400

        signals, score = generate_signals(dc)
        prediction = predict_trend(dc, score)

        # Chart data
        chart_df = dc.tail(252)
        dates = [d.strftime('%Y-%m-%d') for d in chart_df.index]
        def sl(s): return [None if pd.isna(v) else round(float(v), 2) for v in s]

        # Candlestick data
        candles = []
        for i, row in chart_df.iterrows():
            candles.append({
                'time': i.strftime('%Y-%m-%d'),
                'open': round(row['Open'], 2),
                'high': round(row['High'], 2),
                'low': round(row['Low'], 2),
                'close': round(row['Close'], 2),
            })

        vol_data = [{'time': d, 'value': round(float(v), 0), 'color': 'rgba(38,166,154,0.5)' if chart_df['Close'].iloc[i] >= chart_df['Open'].iloc[i] else 'rgba(239,83,80,0.5)'}
                    for i, (d, v) in enumerate(zip(dates, chart_df['Volume']))]

        chart = {
            'dates': dates,
            'candles': candles,
            'volume': vol_data,
            'close': sl(chart_df['Close']),
            'sma20': sl(chart_df['SMA_20']),
            'sma50': sl(chart_df['SMA_50']),
            'sma200': sl(chart_df['SMA_200']),
            'rsi': sl(chart_df['RSI']),
            'macd': sl(chart_df['MACD']),
            'macd_signal': sl(chart_df['MACD_Signal']),
            'macd_hist': sl(chart_df['MACD_Hist']),
            'bb_upper': sl(chart_df['BB_Upper']),
            'bb_mid': sl(chart_df['BB_Mid']),
            'bb_lower': sl(chart_df['BB_Lower']),
            'stoch_k': sl(chart_df['Stoch_K']),
            'stoch_d': sl(chart_df['Stoch_D']),
            'obv': sl(chart_df['OBV']),
            'atr': sl(chart_df['ATR']),
            'vwap': sl(chart_df['VWAP']),
        }

        name = info.get('shortName') or info.get('longName') or ticker

        # Advice
        advice = get_advice(score, prediction)

        return jsonify({
            'ticker': ticker, 'name': name,
            'signals': [{'type': s[0], 'name': s[1], 'desc': s[2]} for s in signals],
            'prediction': prediction, 'advice': advice, 'chart': chart,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/market')
def market_overview():
    """Fetch all major indices, sectors, and key assets."""
    tickers = {
        # Major Indices
        '^GSPC': 'S&P 500', '^DJI': 'Dow Jones', '^IXIC': 'Nasdaq',
        '^RUT': 'Russell 2000', '^VIX': 'VIX (Fear Index)',
        # International
        '^FTSE': 'FTSE 100', '^GDAXI': 'DAX', '^N225': 'Nikkei 225', '^HSI': 'Hang Seng',
        # Sectors (SPDR)
        'XLK': 'Tech', 'XLF': 'Financials', 'XLV': 'Healthcare',
        'XLE': 'Energy', 'XLY': 'Consumer Disc.', 'XLP': 'Consumer Staples',
        'XLI': 'Industrials', 'XLB': 'Materials', 'XLRE': 'Real Estate',
        'XLU': 'Utilities', 'XLC': 'Communication',
        # Commodities / Crypto / Bonds
        'GC=F': 'Gold', 'SI=F': 'Silver', 'CL=F': 'Crude Oil',
        'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum',
        'TLT': 'Long-Term Bonds', 'SHY': 'Short-Term Bonds',
    }

    results = []
    symbols = list(tickers.keys())
    batch_size = 10

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        joined = ' '.join(batch)
        try:
            data = yf.download(joined, period='5d', group_by='ticker', progress=False, threads=True)
            for sym in batch:
                try:
                    if len(batch) == 1:
                        d = data
                    else:
                        d = data[sym] if sym in data.columns.get_level_values(0) else None
                    if d is None or d.empty: continue
                    d = d.dropna()
                    if len(d) < 2: continue
                    last = float(d['Close'].iloc[-1])
                    prev = float(d['Close'].iloc[-2])
                    chg = last - prev
                    chg_pct = (chg / prev) * 100
                    results.append({
                        'symbol': sym, 'name': tickers[sym],
                        'price': round(last, 2),
                        'change': round(chg, 2),
                        'change_pct': round(chg_pct, 2),
                    })
                except: continue
        except: continue

    return jsonify(results)


@app.route('/api/etfs')
def compare_etfs():
    etfs = {
        'VOO':'S&P 500 (Vanguard)','SPY':'S&P 500 (SPDR)','IVV':'S&P 500 (iShares)',
        'QQQ':'Nasdaq 100','VTI':'US Total Market','VXUS':'International',
        'BND':'US Bonds','VNQ':'Real Estate','GLD':'Gold','VIG':'Dividend Growth',
        'SCHD':'Dividend Value','ARKK':'Innovation (ARK)','VGT':'Info Tech','VHT':'Healthcare',
    }
    results = []
    for sym, label in etfs.items():
        try:
            t = yf.Ticker(sym)
            h = t.history(period='1y')
            if h.empty: continue
            ret = (h['Close'].iloc[-1] / h['Close'].iloc[0] - 1) * 100
            vol = h['Close'].pct_change().std() * np.sqrt(252) * 100
            info = {}
            try: info = t.info or {}
            except: pass
            results.append({
                'symbol': sym, 'label': label,
                'price': round(float(h['Close'].iloc[-1]), 2),
                'return_1y': round(ret, 2),
                'volatility': round(vol, 2),
                'sharpe': round(ret / vol, 2) if vol > 0 else 0,
                'expense': info.get('annualReportExpenseRatio') or info.get('expenseRatio') or 'N/A',
                'yield_pct': round(info.get('yield', 0) * 100, 2) if info.get('yield') else 'N/A',
            })
        except: continue
    return jsonify(results)


@app.route('/api/brokers')
def brokers():
    return jsonify(BROKERS)


@app.route('/api/screener')
def screener():
    """Screen popular stocks with key metrics."""
    stocks = ['AAPL','MSFT','NVDA','GOOGL','AMZN','META','TSLA','BRK-B',
              'JPM','V','JNJ','WMT','PG','MA','HD','DIS','NFLX','PYPL',
              'AMD','CRM','INTC','CSCO','PEP','KO','MRK','ABT','COST',
              'NKE','BA','GS','MS','UBER','SQ','SHOP','PLTR','COIN',
              'SOFI','RIVN','ABNB','SNAP','RBLX','DKNG']
    results = []
    joined = ' '.join(stocks)
    try:
        data = yf.download(joined, period='6mo', group_by='ticker', progress=False, threads=True)
        for sym in stocks:
            try:
                d = data[sym].dropna() if sym in data.columns.get_level_values(0) else None
                if d is None or len(d) < 20: continue
                last = float(d['Close'].iloc[-1])
                r1m = (last / float(d['Close'].iloc[-21]) - 1) * 100 if len(d) > 21 else 0
                r3m = (last / float(d['Close'].iloc[-63]) - 1) * 100 if len(d) > 63 else 0
                vol = d['Close'].pct_change().std() * np.sqrt(252) * 100
                rsi_val = float(rsi(d['Close']).iloc[-1])
                sma50_val = float(sma(d['Close'], 50).iloc[-1]) if len(d) >= 50 else last
                above_sma50 = last > sma50_val

                results.append({
                    'symbol': sym, 'price': round(last, 2),
                    'r1m': round(r1m, 2), 'r3m': round(r3m, 2),
                    'vol': round(vol, 1), 'rsi': round(rsi_val, 1),
                    'above_sma50': above_sma50,
                    'trend': 'bullish' if rsi_val > 50 and above_sma50 else 'bearish' if rsi_val < 50 and not above_sma50 else 'neutral',
                })
            except: continue
    except: pass
    return jsonify(results)


@app.route('/api/live')
def live_quote():
    """Lightweight endpoint for live price updates."""
    ticker = request.args.get('ticker', '^GSPC')
    try:
        t = yf.Ticker(ticker)
        h = t.history(period='2d')
        if h.empty:
            return jsonify({'error': 'No data'}), 404
        last = float(h['Close'].iloc[-1])
        prev = float(h['Close'].iloc[-2]) if len(h) > 1 else last
        chg = last - prev
        pct = (chg / prev * 100) if prev else 0
        hi = float(h['High'].iloc[-1])
        lo = float(h['Low'].iloc[-1])
        vol = int(h['Volume'].iloc[-1])
        return jsonify({
            'price': round(last, 2), 'change': round(chg, 2),
            'change_pct': round(pct, 2), 'high': round(hi, 2),
            'low': round(lo, 2), 'volume': vol,
            'time': datetime.now().strftime('%H:%M:%S'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def get_advice(score, pred):
    tips = [
        {'title':'Dollar-Cost Averaging','desc':'Invest a fixed amount on a regular schedule (weekly/monthly). This is the single most effective strategy for beginners — it removes emotion and timing from the equation.','priority':'high'},
        {'title':'Start with Index Funds','desc':'S&P 500 funds (VOO, SPY, IVV) return ~10%/year historically. One purchase = 500 companies. Expense ratios under 0.04%.','priority':'high'},
        {'title':'Tax-Advantaged Accounts First','desc':'Max your Roth IRA ($7,000/yr) and 401k employer match before taxable investing. Tax-free growth is the biggest edge you have.','priority':'high'},
        {'title':'Emergency Fund First','desc':'Keep 3-6 months expenses in a high-yield savings account (5%+ APY) before investing. Never invest money you need within 2 years.','priority':'high'},
        {'title':'The 3-Fund Portfolio','desc':'Simple & proven: VTI (US stocks, 60%) + VXUS (international, 30%) + BND (bonds, 10%). Adjust bond % = your age minus 10.','priority':'high'},
        {'title':'Avoid Individual Stocks Early','desc':'Until you understand financial statements and valuation, stick to diversified funds. Even pros fail to beat the index 80% of the time.','priority':'medium'},
        {'title':'Reinvest Dividends','desc':'Turn on DRIP (dividend reinvestment). Compounding dividends can double your money vs taking cash.','priority':'medium'},
        {'title':'Time > Timing','desc':'$10,000 invested in S&P 500 in 2000 (before the crash) would be $60,000+ today. Missing the 10 best days cuts returns in half.','priority':'medium'},
    ]

    if score > 15:
        tips.insert(1, {'title':'Positive Market Momentum','desc':f'Technical indicators are bullish (score: {score:+d}). Good conditions to start or continue regular investments. Don\'t go all-in — spread purchases over weeks.','priority':'medium'})
    elif score < -15:
        tips.insert(1, {'title':'Market Showing Weakness','desc':f'Technical indicators lean bearish (score: {score:+d}). For long-term investors, dips are buying opportunities. Keep DCA going — buying during fear historically yields the best returns.','priority':'medium'})

    if pred['annual_vol'] > 22:
        tips.append({'title':f'Elevated Volatility ({pred["annual_vol"]:.0f}%)','desc':'Markets are choppier than usual. Consider adding a small bond allocation (BND/AGG) to reduce portfolio swings.','priority':'medium'})

    return tips


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
