"""Invest Today - personal NSE stock analysis dashboard.
Run:  pip install -r requirements.txt  &&  streamlit run app.py
Decision-support only. Not financial advice.
"""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="Invest Today", page_icon="📈", layout="wide")

BENCH = "^NSEI"
DEFAULT_WATCHLIST = (
    "RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, BHARTIARTL, ITC, SBIN, LT, HINDUNILVR, "
    "KOTAKBANK, AXISBANK, BAJFINANCE, MARUTI, SUNPHARMA, TITAN, ASIANPAINT, ULTRACEMCO, "
    "NTPC, POWERGRID, TATASTEEL, M&M, ONGC, COALINDIA, ADANIPORTS, HCLTECH, WIPRO, "
    "TECHM, DRREDDY, CIPLA, EICHERMOT, BAJAJ-AUTO"
)
HORIZONS = {"Short (5 trading days)": 5, "Swing (20 days)": 20, "Positional (60 days)": 60}
BUY_SCORE, MIN_WIN, MIN_SAMPLES = 65, 55, 15


# ---------------------------------------------------------------- data
@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def load(tickers: tuple):
    raw = yf.download(list(tickers), period="6y", auto_adjust=True,
                      group_by="ticker", progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            d = raw[t].dropna(subset=["Close"])
            if len(d) > 260:
                out[t] = d
        except KeyError:
            pass
    return out


# ---------------------------------------------------------------- indicators
def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def features(df, bench):
    c = df["Close"]
    f = pd.DataFrame(index=df.index)
    f["close"] = c
    f["sma50"] = c.rolling(50).mean()
    f["sma200"] = c.rolling(200).mean()
    f["rsi"] = rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macdh"] = macd - macd.ewm(span=9, adjust=False).mean()
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - c.shift()).abs(),
                    (df["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    f["atr"] = tr.rolling(14).mean()
    f["volx"] = df["Volume"] / df["Volume"].rolling(20).mean()
    b = bench["Close"].reindex(c.index).ffill()
    f["rs"] = c.pct_change(60) - b.pct_change(60)

    trend = 15 * (c > f.sma200) + 10 * (c > f.sma50) + 5 * (f.sma50 > f.sma200)
    rsi_pts = pd.Series(np.where((f.rsi >= 50) & (f.rsi <= 70), 20,
                                np.where((f.rsi >= 40) & (f.rsi < 50), 8, 0)), index=c.index)
    macd_pts = 15 * (f.macdh > 0) + 10 * (f.macdh > f.macdh.shift(3))
    vol_pts = 10 * (f.volx > 1.2)
    rs_pts = 15 * (f.rs > 0)
    f["score"] = trend + rsi_pts + macd_pts + vol_pts + rs_pts
    return f


def backtest(f, n):
    """Same rule, applied to history: what happened n days after a BUY-grade score?"""
    fwd = f.close.shift(-n) / f.close - 1
    m = (f.score >= BUY_SCORE) & f.sma200.notna() & fwd.notna()
    r = fwd[m] * 100
    if len(r) < 5:
        return None
    return dict(n=len(r), win=(r > 0).mean() * 100, med=r.median(),
                p25=r.quantile(.25), p75=r.quantile(.75), bad=r.quantile(.10), dist=r)


# ---------------------------------------------------------------- sidebar
page = st.sidebar.radio("View", ["📈 Picks", "💼 Portfolio"], horizontal=True)
st.sidebar.title("⚙️ Settings")
wl = st.sidebar.text_area("Watchlist (NSE symbols)", DEFAULT_WATCHLIST, height=140)
horizon_label = st.sidebar.selectbox("Holding period", list(HORIZONS), index=1)
N = HORIZONS[horizon_label]
capital = st.sidebar.number_input("Capital (₹)", 10000, 100000000, 100000, step=10000)
risk_pct = st.sidebar.slider("Max risk per trade (% of capital)", 0.25, 3.0, 1.0, 0.25)
max_pos = st.sidebar.slider("Max per stock (% of capital)", 5, 50, 20, 5)
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()

symbols = [s.strip().upper() for s in wl.replace("\n", ",").split(",") if s.strip()]
tickers = tuple(dict.fromkeys(s if s.startswith("^") or s.endswith(".NS") else s + ".NS"
                              for s in symbols))
# ---------------------------------------------------------------- portfolio storage
PF_FILE = Path(__file__).with_name("portfolio.csv")
PF_COLS = ["Symbol", "Qty", "Buy price", "Buy date", "Stop ₹"]


def norm(s):
    s = str(s).strip().upper()
    return s if s.startswith("^") or s.endswith(".NS") else s + ".NS"


def load_pf():
    df = pd.DataFrame({"Symbol": pd.Series(dtype="object"), "Qty": pd.Series(dtype="float"),
                       "Buy price": pd.Series(dtype="float"),
                       "Buy date": pd.Series(dtype="datetime64[ns]"),
                       "Stop ₹": pd.Series(dtype="float")})
    if PF_FILE.exists():
        try:
            df = pd.read_csv(PF_FILE, parse_dates=["Buy date"]).reindex(columns=PF_COLS)
        except Exception:
            pass
    return df


if "pf" not in st.session_state:
    st.session_state.pf = load_pf()
pf_tickers = tuple(norm(x) for x in load_pf()["Symbol"].dropna())

data = load(tuple(dict.fromkeys(tickers + pf_tickers)) + (BENCH,))
bench = data.pop(BENCH, None)

st.title("📈 Invest Today")
if bench is None or not data:
    st.error("Could not fetch data. Check your internet connection or symbols.")
    st.stop()

# ---------------------------------------------------------------- market regime
nb = bench["Close"]
regime_up = nb.iloc[-1] > nb.rolling(200).mean().iloc[-1]
nifty_chg = (nb.iloc[-1] / nb.iloc[-2] - 1) * 100

# ---------------------------------------------------------------- analyse
rows, feats, bts = [], {}, {}
for t, d in data.items():
    if t not in tickers:
        continue
    f = features(d, bench)
    last = f.iloc[-1]
    if pd.isna(last.sma200) or pd.isna(last.atr):
        continue
    bt = backtest(f, N)
    feats[t], bts[t] = f, bt
    score = last.score - (10 if not regime_up else 0)  # be pickier in a weak market
    if bt and score >= BUY_SCORE and bt["win"] >= MIN_WIN and bt["med"] > 0 and bt["n"] >= MIN_SAMPLES:
        sig = "BUY"
    elif score >= 50:
        sig = "WATCH"
    else:
        sig = "AVOID"
    price = last.close
    stop = price - 2 * last.atr
    risk_ps = price - stop
    qty = int(max(0, min((capital * risk_pct / 100) // risk_ps, (capital * max_pos / 100) // price)))
    med = bt["med"] if bt else np.nan
    rows.append({
        "Stock": t.replace(".NS", ""), "_t": t, "Price ₹": round(price, 2), "Score": int(score),
        "Signal": sig, "Exp. return %": round(med, 1) if bt else None,
        "Typical range %": f"{bt['p25']:.1f} to {bt['p75']:.1f}" if bt else "n/a",
        "Win rate %": round(bt["win"]) if bt else None,
        "Samples": bt["n"] if bt else 0,
        "Target ₹": round(price * (1 + med / 100), 2) if bt else None,
        "Stop ₹": round(stop, 2), "Qty": qty,
        "Exp. profit ₹": round(qty * price * med / 100) if bt else None,
        "Max loss ₹": round(qty * risk_ps),
    })

res = pd.DataFrame(rows)
res["_r"] = res.Signal.map({"BUY": 0, "WATCH": 1, "AVOID": 2})
res = res.sort_values(["_r", "Exp. return %"], ascending=[True, False]).drop(columns="_r").reset_index(drop=True)

# ---------------------------------------------------------------- portfolio view
if page.endswith("Portfolio"):
    st.subheader("💼 My portfolio")
    st.caption("Add or edit holdings below. Saved automatically to portfolio.csv next to app.py.")
    edited = st.data_editor(
        st.session_state.pf, num_rows="dynamic", use_container_width=True, hide_index=True,
        key="pf_editor",
        column_config={
            "Symbol": st.column_config.TextColumn(help="NSE symbol, e.g. TCS"),
            "Qty": st.column_config.NumberColumn(min_value=1, step=1),
            "Buy price": st.column_config.NumberColumn("Buy price ₹", min_value=0.0, format="%.2f"),
            "Buy date": st.column_config.DateColumn(),
            "Stop ₹": st.column_config.NumberColumn(help="Optional. Blank = auto (2x ATR below buy)",
                                                    min_value=0.0),
        })
    clean = edited.dropna(subset=["Symbol", "Qty", "Buy price"]).copy()
    clean["Symbol"] = clean["Symbol"].map(norm)
    clean.to_csv(PF_FILE, index=False)

    if clean.empty:
        st.info("No holdings yet. Add a row above (symbol, quantity, buy price).")
        st.stop()

    prow, missing = [], []
    for _, r in clean.iterrows():
        t = r["Symbol"]
        if t not in data:
            missing.append(t)
            continue
        f = feats.get(t)
        if f is None:
            f = features(data[t], bench)
        last, prev = f.iloc[-1], f.close.iloc[-2]
        price, qty, buy, atr = last.close, r["Qty"], r["Buy price"], last.atr
        stop = r["Stop ₹"] if pd.notna(r["Stop ₹"]) and r["Stop ₹"] > 0 else buy - 2 * atr
        invested, value = qty * buy, qty * price
        pnl = value - invested
        if pd.notna(stop) and price <= stop:
            act = "🔴 EXIT (stop hit)"
        elif last.score < 40 or (price < last.sma50 and last.macdh < 0):
            act = "🟠 REVIEW (trend weak)"
        elif pnl > 0 and last.rsi > 75:
            act = "🟡 BOOK PARTIAL (overbought)"
        else:
            act = "🟢 HOLD"
        trail = max(stop, price - 2 * atr) if pnl > 0 and pd.notna(atr) else stop
        held = (date.today() - pd.Timestamp(r["Buy date"]).date()).days if pd.notna(r["Buy date"]) else None
        prow.append({"Stock": t.replace(".NS", ""), "Qty": int(qty), "Buy ₹": round(buy, 2),
                     "Now ₹": round(price, 2), "Invested ₹": round(invested), "Value ₹": round(value),
                     "P&L ₹": round(pnl), "P&L %": round(pnl / invested * 100, 1),
                     "Today %": round((price / prev - 1) * 100, 2),
                     "Today ₹": round(qty * (price - prev)), "Days held": held,
                     "Score": int(last.score), "Stop ₹": round(stop, 2) if pd.notna(stop) else None,
                     "Suggested stop ₹": round(trail, 2) if pd.notna(trail) else None, "Action": act})

    fresh = [m for m in missing if m not in st.session_state.setdefault("tried", set())]
    if fresh:  # new symbol: fetch its data once
        st.session_state["tried"].update(fresh)
        st.rerun()
    if missing:
        st.warning("No price data for: " + ", ".join(m.replace(".NS", "") for m in missing))
    if not prow:
        st.stop()

    pdf = pd.DataFrame(prow)
    pdf["Weight %"] = (pdf["Value ₹"] / pdf["Value ₹"].sum() * 100).round(1)
    inv, val = pdf["Invested ₹"].sum(), pdf["Value ₹"].sum()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Invested", f"₹{inv:,.0f}")
    m2.metric("Current value", f"₹{val:,.0f}")
    m3.metric("Total P&L", f"₹{val - inv:,.0f}", f"{(val / inv - 1) * 100:+.1f}%")
    m4.metric("Today", f"₹{pdf['Today ₹'].sum():,.0f}")

    exits = pdf[pdf.Action.str.contains("EXIT")]["Stock"].tolist()
    if exits:
        st.error("Stop-loss hit: " + ", ".join(exits) + ". Your plan says exit.")
    if pdf["Weight %"].max() > 30:
        big = pdf.loc[pdf["Weight %"].idxmax(), "Stock"]
        st.warning(f"{big} is over 30% of your portfolio. Consider reducing concentration.")
    if not regime_up:
        st.warning("Nifty is below its 200-day average. Be careful adding new money.")

    col = lambda v: "color:#16a34a" if isinstance(v, (int, float)) and v > 0 else (
        "color:#dc2626" if isinstance(v, (int, float)) and v < 0 else "")
    st.dataframe(pdf.style.map(col, subset=["P&L ₹", "P&L %", "Today %", "Today ₹"]),
                 use_container_width=True, hide_index=True)

    pie = go.Figure(go.Pie(labels=pdf["Stock"], values=pdf["Value ₹"], hole=.5))
    pie.update_layout(height=320, title="Allocation", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(pie, use_container_width=True)
    st.caption("Actions are rule-based hints (stop hit, weak trend, overbought), not advice. "
               "Suggested stop trails up only when you're in profit.")
    st.stop()

# ---------------------------------------------------------------- KPIs
buys = res[res.Signal == "BUY"]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Nifty 50", f"{nb.iloc[-1]:,.0f}", f"{nifty_chg:+.2f}%")
k2.metric("Market trend", "Uptrend ✅" if regime_up else "Weak ⚠️",
          "above 200-DMA" if regime_up else "below 200-DMA", delta_color="off")
k3.metric("BUY signals today", len(buys))
k4.metric("Holding period", f"{N} days")
if not regime_up:
    st.warning("Nifty is below its 200-day average. Scores are reduced; consider smaller positions.")

# ---------------------------------------------------------------- picks
st.subheader("🎯 Today's picks")
if buys.empty:
    st.info("No stock passes all checks today. Staying in cash is a valid decision.")
show = res.drop(columns="_t")
st.dataframe(
    show.style.map(lambda v: {"BUY": "background-color:#14532d;color:white",
                              "WATCH": "background-color:#854d0e;color:white",
                              "AVOID": "color:#888"}.get(v, ""), subset=["Signal"]),
    use_container_width=True, hide_index=True)
if len(buys):
    total_cost = (buys["Qty"] * buys["Price ₹"]).sum()
    st.caption(f"If you took every BUY at the suggested size: invest ≈ ₹{total_cost:,.0f}, "
               f"expected profit ≈ ₹{buys['Exp. profit ₹'].sum():,.0f} in ~{N} trading days "
               f"(historical median, not a promise), max planned loss ≈ ₹{buys['Max loss ₹'].sum():,.0f}.")

# ---------------------------------------------------------------- detail
st.subheader("🔍 Stock detail")
pick = st.selectbox("Choose stock", res["Stock"].tolist())
t = pick + ".NS"
f, bt = feats[t], bts[t]
row = res[res.Stock == pick].iloc[0]
view = f.tail(250)

c1, c2 = st.columns([2, 1])
with c1:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[.6, .2, .2], vertical_spacing=.03)
    d = data[t].loc[view.index]
    fig.add_trace(go.Candlestick(x=view.index, open=d.Open, high=d.High, low=d.Low, close=d.Close,
                                 name="Price"), 1, 1)
    fig.add_trace(go.Scatter(x=view.index, y=view.sma50, name="SMA50", line=dict(width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=view.index, y=view.sma200, name="SMA200", line=dict(width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=view.index, y=view.rsi, name="RSI", line=dict(width=1)), 2, 1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)
    fig.add_trace(go.Bar(x=view.index, y=view.macdh, name="MACD hist"), 3, 1)
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=10, b=10))
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown(f"### {pick}: **{row.Signal}**")
    st.metric("Score", f"{row.Score}/100")
    st.markdown(
        f"**Entry:** ₹{row['Price ₹']:,}  \n"
        f"**Target:** ₹{row['Target ₹']}  \n"
        f"**Stop-loss:** ₹{row['Stop ₹']:,}  \n"
        f"**Suggested qty:** {row['Qty']}  \n"
        f"**Expected profit:** ₹{row['Exp. profit ₹']}  \n"
        f"**Max loss:** ₹{row['Max loss ₹']:,}")
    if bt:
        st.caption(f"In {bt['n']} similar past setups, a {N}-day hold won {bt['win']:.0f}% of the time. "
                   f"Worst 10% of outcomes: {bt['bad']:.1f}%.")
        h = go.Figure(go.Histogram(x=bt["dist"], nbinsx=30))
        h.add_vline(x=0, line_dash="dash")
        h.update_layout(height=260, title=f"{N}-day return after similar signals (%)",
                        margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(h, use_container_width=True)
    else:
        st.caption("Not enough history of similar setups for a reliable estimate.")

st.divider()
st.caption("Decision-support only, not financial advice. Backtests use the same fixed rules on past "
           "prices; overlapping signals mean the real sample is smaller than it looks, and past "
           "results never guarantee future returns. Always use the stop-loss.")
