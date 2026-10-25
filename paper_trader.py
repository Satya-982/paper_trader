"""Paper Trader - practice investing with virtual money, using Stock Market Advisor signals.
Run:  pip install -r requirements.txt  &&  streamlit run paper_trader.py
Practice tool only. Not financial advice. No real money is involved.
"""
import json
import re
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Paper Trader", page_icon="🎮", layout="wide")

IST = timezone(timedelta(hours=5, minutes=30))
BENCH = "^NSEI"
IS_CLOUD = Path("/mount/src").exists()  # Streamlit Community Cloud
SAVE_FILE = Path(__file__).with_name("paper_account.json")
BUY_SCORE, MIN_WIN, MIN_SAMPLES = 65, 55, 15
START_CASH = 100000
DEFAULT_WATCHLIST = (
    "RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, BHARTIARTL, ITC, SBIN, LT, HINDUNILVR, "
    "KOTAKBANK, AXISBANK, BAJFINANCE, MARUTI, SUNPHARMA, TITAN, ASIANPAINT, ULTRACEMCO, "
    "NTPC, POWERGRID, TATASTEEL, M&M, ONGC, COALINDIA, ADANIPORTS, HCLTECH, WIPRO, "
    "TECHM, DRREDDY, CIPLA, EICHERMOT, BAJAJ-AUTO"
)
HORIZONS = {"Short (5 trading days)": 5, "Swing (20 days)": 20, "Positional (60 days)": 60}


# ================================================================ small helpers
def inr(x):
    """Indian digit grouping, e.g. 1,00,000."""
    neg = x < 0
    s = str(int(round(abs(x))))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        s = head + "," + tail
    return ("-" if neg else "") + "₹" + s


def px_fmt(x):
    return f"₹{x:,.2f}"


def norm(s):
    s = str(s).strip().upper()
    return s if s.startswith("^") or s.endswith(".NS") else s + ".NS"


def plain(sym):
    return sym.replace(".NS", "")


def now_iso():
    return datetime.now(IST).isoformat(timespec="seconds")


def market_status():
    """NSE regular session: Mon-Fri 09:15-15:30 IST (public holidays not tracked)."""
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False, "Closed (weekend)"
    if dtime(9, 15) <= now.time() <= dtime(15, 30):
        return True, "Open"
    return False, "Closed"


def bucket():
    open_, _ = market_status()
    return int(datetime.now().timestamp() // (30 if open_ else 900))


# ================================================================ signal engine (same rules as the Advisor app)
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
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - c.shift()).abs(),
                    (df["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    f["atr"] = tr.rolling(14).mean()
    f["volx"] = df["Volume"] / df["Volume"].rolling(20).mean()
    b = bench["Close"].reindex(c.index).ffill()
    f["rs"] = c.pct_change(60) - b.pct_change(60)
    trend = 15 * (c > f.sma200) + 10 * (c > f.sma50) + 5 * (f.sma50 > f.sma200)
    rsi_pts = pd.Series(np.where((f.rsi >= 50) & (f.rsi <= 70), 20,
                                 np.where((f.rsi >= 40) & (f.rsi < 50), 8, 0)), index=c.index)
    macd_pts = 15 * (f.macdh > 0) + 10 * (f.macdh > f.macdh.shift(3))
    f["score"] = trend + rsi_pts + macd_pts + 10 * (f.volx > 1.2) + 15 * (f.rs > 0)
    return f


def backtest(f, n):
    fwd = f.close.shift(-n) / f.close - 1
    m = (f.score >= BUY_SCORE) & f.sma200.notna() & fwd.notna()
    r = fwd[m] * 100
    if len(r) < 5:
        return None
    return dict(n=len(r), win=(r > 0).mean() * 100, med=r.median())


@st.cache_data(ttl=3600, show_spinner="Loading market history...")
def load_history(tickers: tuple):
    out = {}
    try:
        raw = yf.download(list(tickers), period="6y", auto_adjust=True,
                          group_by="ticker", progress=False, threads=True)
    except Exception:
        return out
    for t in tickers:
        try:
            d = raw[t].dropna(subset=["Close"])
            if len(d) > 260:
                out[t] = d
        except (KeyError, TypeError):
            pass
    return out


@st.cache_data(ttl=3600, show_spinner="Analysing the market...")
def get_signals(tickers: tuple, N: int, equity_b: int, risk_pct: float, max_pos: int):
    data = load_history(tickers + (BENCH,))
    bench = data.pop(BENCH, None)
    if bench is None or not data:
        return pd.DataFrame(), True
    nb = bench["Close"]
    regime_up = bool(nb.iloc[-1] > nb.rolling(200).mean().iloc[-1])
    rows = []
    for t, d in data.items():
        f = features(d, bench)
        last = f.iloc[-1]
        if pd.isna(last.sma200) or pd.isna(last.atr):
            continue
        bt = backtest(f, N)
        score = int(last.score - (0 if regime_up else 10))
        if bt and score >= BUY_SCORE and bt["win"] >= MIN_WIN and bt["med"] > 0 and bt["n"] >= MIN_SAMPLES:
            sig = "BUY"
        elif score >= 50:
            sig = "WATCH"
        else:
            sig = "AVOID"
        price = float(last.close)
        stop = price - 2 * float(last.atr)
        risk_ps = price - stop
        qty = int(max(0, min((equity_b * risk_pct / 100) // risk_ps, (equity_b * max_pos / 100) // price)))
        med = bt["med"] if bt else np.nan
        rows.append({"Stock": plain(t), "_t": t, "Close": round(price, 2), "Score": score, "Signal": sig,
                     "Exp %": round(med, 1) if bt else None, "Win %": round(bt["win"]) if bt else None,
                     "Samples": bt["n"] if bt else 0,
                     "Target": round(price * (1 + med / 100), 2) if bt else None,
                     "Stop": round(stop, 2), "Qty": qty})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, regime_up
    df["_r"] = df.Signal.map({"BUY": 0, "WATCH": 1, "AVOID": 2})
    df = df.sort_values(["_r", "Exp %"], ascending=[True, False]).drop(columns="_r").reset_index(drop=True)
    return df, regime_up


# ================================================================ live prices
def _close(raw, t):
    try:
        s = raw[t]["Close"]
    except (KeyError, TypeError):
        try:
            s = raw["Close"][t]
        except (KeyError, TypeError, IndexError):
            return None
    return s.dropna()


def fmt_ts(ts):
    try:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("Asia/Kolkata")
        if ts.hour == 0 and ts.minute == 0:
            return ts.strftime("%d %b (close)")
        return ts.strftime("%d %b %H:%M")
    except Exception:
        return ""


@st.cache_data(ttl=3600, show_spinner=False)
def live_prices(tickers: tuple, bucket_id: int):
    """Latest traded price per ticker. Yahoo's NSE feed can lag a few minutes."""
    tickers = tuple(dict.fromkeys(tickers))
    out = {}

    def grab(period, interval):
        try:
            raw = yf.download(list(tickers), period=period, interval=interval,
                              group_by="ticker", progress=False, threads=True)
        except Exception:
            return
        for t in tickers:
            if t in out:
                continue
            s = _close(raw, t)
            if s is not None and len(s):
                out[t] = (float(s.iloc[-1]), fmt_ts(s.index[-1]))

    grab("1d", "1m")
    if len(out) < len(tickers):
        grab("5d", "1d")
    return out


# ================================================================ account
def new_account(cash):
    return {"start_cash": float(cash), "cash": float(cash), "created": now_iso(),
            "bench_start": None, "positions": {}, "journal": [], "equity": []}


def save(a):
    if IS_CLOUD:
        return
    try:
        SAVE_FILE.write_text(json.dumps(a), encoding="utf-8")
    except Exception:
        pass


def load_saved():
    if not IS_CLOUD and SAVE_FILE.exists():
        try:
            a = json.loads(SAVE_FILE.read_text(encoding="utf-8"))
            if {"cash", "positions", "journal", "start_cash"} <= a.keys():
                a.setdefault("equity", [])
                a.setdefault("bench_start", None)
                return a
        except Exception:
            pass
    return None


def do_buy(a, sym, qty, price, stop, target, fee_rate, note="Buy"):
    qty = int(qty)
    if qty < 1:
        return False, "Quantity must be at least 1."
    if stop and stop >= price:
        return False, f"Stop-loss ({px_fmt(stop)}) must be below the buy price ({px_fmt(price)})."
    if target and target <= price:
        return False, f"Target ({px_fmt(target)}) must be above the buy price ({px_fmt(price)})."
    gross = price * qty
    fee = gross * fee_rate
    cost = gross + fee
    if cost > a["cash"] + 1e-9:
        return False, f"Not enough cash. Need {inr(cost)}, you have {inr(a['cash'])}."
    pos = a["positions"].get(sym)
    if pos:
        tot = pos["qty"] + qty
        pos["avg"] = (pos["qty"] * pos["avg"] + cost) / tot  # cost basis includes charges
        pos["qty"] = tot
    else:
        pos = {"qty": qty, "avg": cost / qty, "stop": None, "target": None, "opened": now_iso()}
        a["positions"][sym] = pos
    if stop and stop > 0:
        pos["stop"] = float(stop)
    if target and target > 0:
        pos["target"] = float(target)
    a["cash"] -= cost
    a["journal"].append({"t": now_iso(), "sym": sym, "side": "BUY", "qty": qty, "price": round(price, 2),
                         "fee": round(fee, 2), "pnl": None, "note": note})
    save(a)
    return True, f"Bought {qty} {plain(sym)} at {px_fmt(price)} (charges {px_fmt(fee)})."


def do_sell(a, sym, qty, price, fee_rate, note="Sell"):
    pos = a["positions"].get(sym)
    qty = int(qty)
    if not pos:
        return False, f"You don't hold {plain(sym)}."
    if qty < 1 or qty > pos["qty"]:
        return False, f"You can sell 1 to {pos['qty']} shares of {plain(sym)}."
    gross = price * qty
    fee = gross * fee_rate
    net = gross - fee
    pnl = net - qty * pos["avg"]
    a["cash"] += net
    pos["qty"] -= qty
    if pos["qty"] == 0:
        del a["positions"][sym]
    a["journal"].append({"t": now_iso(), "sym": sym, "side": "SELL", "qty": qty, "price": round(price, 2),
                         "fee": round(fee, 2), "pnl": round(pnl, 2), "note": note})
    save(a)
    return True, f"Sold {qty} {plain(sym)} at {px_fmt(price)}. Realised P&L {inr(pnl)}."


def check_exits(a, prices, fee_rate, open_):
    """Auto-sell positions whose stop-loss or target is touched. Runs only while the market is open
    and the app is open in a browser."""
    events = []
    if not open_:
        return events
    for sym, pos in list(a["positions"].items()):
        p = prices.get(sym)
        if not p:
            continue
        ltp = p[0]
        if pos.get("stop") and ltp <= pos["stop"]:
            note = "Stop-loss hit"
        elif pos.get("target") and ltp >= pos["target"]:
            note = "Target hit"
        else:
            continue
        ok, msg = do_sell(a, sym, pos["qty"], ltp, fee_rate, note)
        if ok:
            events.append(f"{note}: {msg}")
    return events


def valuation(a, prices):
    rows, mkt, cost = [], 0.0, 0.0
    for sym, pos in a["positions"].items():
        p = prices.get(sym)
        ltp = p[0] if p else pos["avg"]
        value = ltp * pos["qty"]
        basis = pos["avg"] * pos["qty"]
        mkt += value
        cost += basis
        rows.append({"Stock": plain(sym), "Qty": pos["qty"], "Avg ₹": round(pos["avg"], 2),
                     "LTP ₹": round(ltp, 2) if p else None, "Value ₹": round(value),
                     "P&L ₹": round(value - basis), "P&L %": round((value / basis - 1) * 100, 2),
                     "Stop ₹": pos.get("stop"), "Target ₹": pos.get("target"),
                     "Price time": p[1] if p else "unavailable"})
    return rows, mkt, cost


def snapshot(a, equity, bench_px, open_):
    now = datetime.now(IST)
    last = a["equity"][-1] if a["equity"] else None
    due = last is None or (
        (now - datetime.fromisoformat(last["t"])).total_seconds() >= 60
        and (open_ or abs(equity - last["v"]) > 0.5))
    if due:
        a["equity"].append({"t": now_iso(), "v": round(equity, 2), "b": bench_px})
        del a["equity"][:-3000]
        save(a)


def green_red(v):
    if isinstance(v, (int, float)) and not pd.isna(v):
        return "color:#16a34a" if v > 0 else ("color:#dc2626" if v < 0 else "")
    return ""


# ================================================================ state + sidebar
if "acct" not in st.session_state:
    st.session_state.acct = load_saved() or new_account(START_CASH)
A = st.session_state.acct

st.sidebar.title("⚙️ Settings")
wl = st.sidebar.text_area("Watchlist (NSE symbols)", DEFAULT_WATCHLIST, height=120)
horizon_label = st.sidebar.selectbox("Signal holding period", list(HORIZONS), index=1)
N = HORIZONS[horizon_label]
risk_pct = st.sidebar.slider("Max risk per trade (% of account)", 0.25, 3.0, 1.0, 0.25)
max_pos = st.sidebar.slider("Max per stock (% of account)", 5, 50, 20, 5)
fee_pct = st.sidebar.number_input("Trading charges per order (%)", 0.0, 1.0, 0.1, 0.05,
                                  help="Rough estimate of brokerage + taxes. Set 0 to ignore.")
st.session_state.fee_rate = fee_pct / 100

with st.sidebar.expander("🗂 Account: reset / backup"):
    new_amt = st.number_input("Starting money (₹)", 100000, 100000000, START_CASH, 50000)
    sure = st.checkbox("I understand this erases all my trades")
    if st.button("Reset account", disabled=not sure):
        st.session_state.acct = new_account(new_amt)
        save(st.session_state.acct)
        st.rerun()
    st.download_button("⬇️ Download backup (JSON)", json.dumps(A), "paper_account.json", "application/json")
    up = st.file_uploader("Restore from backup", type="json")
    if up is not None and st.button("Restore this backup"):
        try:
            b = json.load(up)
            assert {"cash", "positions", "journal", "start_cash"} <= b.keys()
            b.setdefault("equity", [])
            b.setdefault("bench_start", None)
            st.session_state.acct = b
            save(b)
            st.rerun()
        except Exception:
            st.error("That file isn't a valid Paper Trader backup.")
    if IS_CLOUD:
        st.caption("Online version: your account lives in this browser session only. "
                   "Download a backup to keep it.")
    else:
        st.caption("Auto-saved to paper_account.json next to this app.")

tickers = tuple(dict.fromkeys(norm(s) for s in wl.replace("\n", ",").split(",") if s.strip()))

# ================================================================ header
open_now, status_txt = market_status()
st.title("🎮 Paper Trader")
st.caption(f"{'🟢' if open_now else '🔴'} NSE {status_txt} · {datetime.now(IST).strftime('%a %d %b, %H:%M')} IST "
           "· Practice with virtual money at real market prices. No real money involved.")

if "flash" in st.session_state:
    ok, msg = st.session_state.pop("flash")
    (st.success if ok else st.error)(msg)


# ================================================================ live panel (auto-refreshes)
@st.fragment(run_every=30)
def live_panel():
    a = st.session_state.acct
    fee = st.session_state.get("fee_rate", 0.001)
    open_, _ = market_status()
    prices = live_prices(tuple(a["positions"]) + (BENCH,), bucket())
    if a.get("bench_start") is None and prices.get(BENCH):
        a["bench_start"] = prices[BENCH][0]
        save(a)
    for e in check_exits(a, prices, fee, open_):
        st.toast(e, icon="🔔")

    rows, mkt, cost = valuation(a, prices)
    equity = a["cash"] + mkt
    realised = sum(j["pnl"] for j in a["journal"] if j.get("pnl") is not None)
    ret = (equity / a["start_cash"] - 1) * 100
    nb = prices.get(BENCH)
    nret = (nb[0] / a["bench_start"] - 1) * 100 if nb and a.get("bench_start") else None
    snapshot(a, equity, nb[0] if nb else None, open_)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Account value", inr(equity), f"{inr(equity - a['start_cash'])} ({ret:+.2f}%)")
    m2.metric("Cash", inr(a["cash"]))
    m3.metric("Open P&L", inr(mkt - cost), f"{(mkt / cost - 1) * 100:+.2f}%" if cost else None)
    m4.metric("Booked P&L", inr(realised))
    m5.metric("You vs Nifty", f"{ret:+.2f}%",
              f"{ret - nret:+.2f}% vs Nifty ({nret:+.2f}%)" if nret is not None else None,
              delta_color="normal")

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df.style.map(green_red, subset=["P&L ₹", "P&L %"]),
                     width="stretch", hide_index=True)
        if not open_:
            for sym, pos in a["positions"].items():
                p = prices.get(sym)
                if p and pos.get("stop") and p[0] <= pos["stop"]:
                    st.warning(f"{plain(sym)} closed below its stop. It will be sold when the market is "
                               "open and this page is open.")
        c1, c2 = st.columns([3, 1])
        pick = c1.selectbox("Close a position", list(a["positions"]), format_func=plain,
                            key="close_pick", label_visibility="collapsed")
        if c2.button("Sell all at market", key="close_btn", width="stretch"):
            p = prices.get(pick)
            if p:
                ok, msg = do_sell(a, pick, a["positions"][pick]["qty"], p[0], fee, "Manual close")
                st.toast(msg, icon="✅" if ok else "⚠️")
                st.rerun(scope="fragment")
            else:
                st.toast("No price available right now.", icon="⚠️")
        st.caption("Prices refresh every 30 s while the market is open (Yahoo data can lag a few minutes). "
                   "P&L is after buy charges, before sell charges. Stop-loss and target exits fire only "
                   "while this page is open.")
    else:
        st.info("No open positions yet. Open the **Signals** tab, pick a BUY, then place it in the **Trade** tab.")


live_panel()

tab_trade, tab_sig, tab_journal, tab_perf = st.tabs(["💹 Trade", "🎯 Signals", "📒 Journal", "📊 Performance"])

# ---------------------------------------------------------------- signals data
equity_est = A["cash"] + sum(p["qty"] * p["avg"] for p in A["positions"].values())
sig, regime_up = get_signals(tickers, N, int(round(equity_est / 10000) * 10000), risk_pct, max_pos)
sig_map = {r["Stock"]: r for _, r in sig.iterrows()} if not sig.empty else {}

# ---------------------------------------------------------------- trade tab
with tab_trade:
    options = sig["Stock"].tolist() if not sig.empty else [plain(t) for t in tickers]
    left, right = st.columns(2)
    with left:
        choice = st.selectbox(
            "Stock (BUY signals first)", options,
            format_func=lambda s: f"{s}  •  {sig_map[s]['Signal']} ({sig_map[s]['Score']})" if s in sig_map else s)
        custom = st.text_input("...or type any NSE symbol", placeholder="e.g. IRCTC")
        sym_plain = (custom.strip().upper() or (choice or "")).replace(".NS", "")
        side = st.radio("Action", ["Buy", "Sell"], horizontal=True)

    if not sym_plain:
        st.info("Choose or type a stock to trade.")
    else:
        sym = norm(sym_plain)
        srow = sig_map.get(sym_plain)
        quote = live_prices((sym, BENCH), bucket()).get(sym)
        held = A["positions"].get(sym)
        fee_rate = st.session_state.fee_rate
        with left:
            if srow is not None:
                st.caption(f"Signal: **{srow['Signal']}** · score {srow['Score']}/100"
                           + (f" · past {N}-day median {srow['Exp %']:+.1f}%, win rate {srow['Win %']}% "
                              f"({srow['Samples']} samples)" if srow["Exp %"] is not None else ""))
            if quote is None:
                st.error("No live price found for this symbol. Check the spelling.")
            else:
                if side == "Buy":
                    sugg = int(srow["Qty"]) if srow is not None else 0
                    qty = st.number_input("Quantity", 1, 10_000_000, max(1, sugg), key=f"q_{sym}_b",
                                          help="Pre-filled with the risk-sized quantity from the signal.")
                    stop = st.number_input("Stop-loss ₹ (0 = none)", 0.0, 10_000_000.0,
                                           float(srow["Stop"]) if srow is not None else 0.0, 0.5,
                                           format="%.2f", key=f"s_{sym}")
                    tgt = st.number_input("Target ₹ (0 = none)", 0.0, 10_000_000.0,
                                          float(srow["Target"]) if srow is not None and srow["Target"] else 0.0,
                                          0.5, format="%.2f", key=f"t_{sym}")
                elif held:
                    qty = st.number_input("Quantity to sell", 1, int(held["qty"]), int(held["qty"]),
                                          key=f"q_{sym}_s")
                else:
                    qty = 0
                    st.info(f"You don't hold {sym_plain}, so there's nothing to sell.")

        with right:
            if quote is not None:
                price = quote[0]
                st.metric(f"{sym_plain} live price", px_fmt(price), f"as of {quote[1]}", delta_color="off")
                if not open_now:
                    st.warning("Market is closed. Orders fill at the last available price (practice only).")
                if side == "Buy":
                    gross = price * qty
                    fee = gross * fee_rate
                    st.markdown(
                        f"**Order value:** {inr(gross)}  \n**Charges:** {px_fmt(fee)}  \n"
                        f"**Cash after:** {inr(A['cash'] - gross - fee)}  \n"
                        f"**Share of account:** {gross / max(equity_est, 1) * 100:.0f}%")
                    if stop > 0:
                        st.markdown(f"**Loss if stop hits:** {inr(qty * (price - stop))} "
                                    f"({qty * (price - stop) / max(equity_est, 1) * 100:.1f}% of account)")
                    else:
                        st.warning("No stop-loss set. Practise the habit: always set one.")
                    if gross / max(equity_est, 1) * 100 > max_pos:
                        st.warning(f"This is above your {max_pos}% per-stock limit.")
                    if st.button("Place BUY order", type="primary", width="stretch"):
                        ok, msg = do_buy(A, sym, qty, price, stop, tgt, fee_rate,
                                         f"Signal {srow['Signal']}" if srow is not None else "Manual buy")
                        st.session_state.flash = (ok, msg)
                        st.rerun()
                elif held:
                    gross = price * qty
                    fee = gross * fee_rate
                    est = gross - fee - qty * held["avg"]
                    st.markdown(f"**Proceeds:** {inr(gross - fee)} (charges {px_fmt(fee)})  \n"
                                f"**Your average cost:** {px_fmt(held['avg'])}  \n"
                                f"**P&L on this sale:** {inr(est)}")
                    if st.button("Place SELL order", type="primary", width="stretch"):
                        ok, msg = do_sell(A, sym, qty, price, fee_rate, "Manual sell")
                        st.session_state.flash = (ok, msg)
                        st.rerun()

# ---------------------------------------------------------------- signals tab
with tab_sig:
    if sig.empty:
        st.warning("Could not load market data right now. Check your internet connection and try again.")
    else:
        if not regime_up:
            st.warning("Nifty is below its 200-day average, so scores are reduced. Trade smaller.")
        live = live_prices(tuple(sig["_t"]) + (BENCH,), bucket())
        show = sig.drop(columns="_t").copy()
        show.insert(2, "Live ₹", [round(live[t][0], 2) if t in live else None for t in sig["_t"]])
        show["Move %"] = [round((live[t][0] / c - 1) * 100, 2) if t in live else None
                          for t, c in zip(sig["_t"], sig["Close"])]
        st.dataframe(
            show.style.map(lambda v: {"BUY": "background-color:#14532d;color:white",
                                      "WATCH": "background-color:#854d0e;color:white",
                                      "AVOID": "color:#888"}.get(v, ""), subset=["Signal"])
            .map(green_red, subset=["Move %"]),
            width="stretch", hide_index=True)
        st.caption("Signals come from the daily close (same rules as the Stock Market Advisor). Target and "
                   "stop are price levels from that close. 'Move %' is today's change since that close. "
                   "Qty is sized so a stop-out costs about your max-risk %.")

# ---------------------------------------------------------------- journal tab
with tab_journal:
    J = A["journal"]
    if not J:
        st.info("Your trades will appear here.")
    else:
        sells = [j for j in J if j["side"] == "SELL"]
        wins = [j["pnl"] for j in sells if j["pnl"] > 0]
        losses = [j["pnl"] for j in sells if j["pnl"] <= 0]
        j1, j2, j3, j4 = st.columns(4)
        j1.metric("Closed trades", len(sells))
        j2.metric("Win rate", f"{len(wins) / len(sells) * 100:.0f}%" if sells else "n/a")
        j3.metric("Avg win / loss", f"{inr(np.mean(wins)) if wins else '-'} / {inr(np.mean(losses)) if losses else '-'}")
        j4.metric("Charges paid", inr(sum(j["fee"] for j in J)))
        jd = pd.DataFrame([{"Time": datetime.fromisoformat(j["t"]).strftime("%d %b %H:%M"),
                            "Stock": plain(j["sym"]), "Side": j["side"], "Qty": j["qty"],
                            "Price ₹": j["price"], "Charges ₹": j["fee"], "P&L ₹": j["pnl"],
                            "Note": j["note"]} for j in reversed(J)])
        st.dataframe(jd.style.map(green_red, subset=["P&L ₹"]), width="stretch", hide_index=True)
        st.download_button("⬇️ Download journal (CSV)", jd.to_csv(index=False), "trade_journal.csv", "text/csv")


# ---------------------------------------------------------------- performance tab
@st.fragment(run_every=30)
def perf_panel():
    a = st.session_state.acct
    eq = a["equity"]
    if len(eq) < 2:
        st.info("Your equity curve builds up as you keep the page open during market hours.")
        return
    t = [datetime.fromisoformat(e["t"]) for e in eq]
    mine = [(e["v"] / a["start_cash"] - 1) * 100 for e in eq]
    fig = go.Figure(go.Scatter(x=t, y=mine, name="Your account", line=dict(width=2.5)))
    if a.get("bench_start"):
        bench = [((e["b"] / a["bench_start"] - 1) * 100) if e.get("b") else None for e in eq]
        fig.add_trace(go.Scatter(x=t, y=bench, name="Nifty 50", line=dict(width=1.5, dash="dot"),
                                 connectgaps=True))
    fig.add_hline(y=0, line_dash="dash", line_width=1)
    fig.update_layout(height=380, title="Return since you started (%)", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    peak = max(e["v"] for e in eq)
    dd = min((e["v"] / max(x["v"] for x in eq[:i + 1]) - 1) * 100 for i, e in enumerate(eq))
    c1, c2, c3 = st.columns(3)
    c1.metric("Peak value", inr(peak))
    c2.metric("Max drawdown", f"{dd:.1f}%")
    c3.metric("Started", datetime.fromisoformat(a["created"]).strftime("%d %b %Y"))


with tab_perf:
    perf_panel()

st.divider()
st.caption("Practice tool only, not financial advice. Real markets add slippage, delays and emotions that "
           "paper trading can't fully copy. Treat results as training, not proof.")
