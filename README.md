# Stock Market Advisor + Paper Trader

Educational stock-market tools for NSE stocks. Not financial advice. No real money is used.

## What's inside

| File | What it is |
|---|---|
| `index.html` (+ `manifest.webmanifest`, `sw.js`, icons) | **Paper Trader web app.** Runs entirely in the browser on GitHub Pages and can be installed like an app. Practice with Rs. 1,00,000 of virtual money at live NSE prices. |
| `app.py` | **Stock Market Advisor** (Streamlit). Daily ranked picks, expected return, win rate, target, stop-loss, portfolio view. |
| `paper_trader.py` | **Paper Trader** (Streamlit version of the web app). |
| `requirements.txt` | Python packages for the Streamlit apps. |

## Run the web app

1. Repo **Settings > Pages**, set **Source** to *Deploy from a branch*, branch `main`, folder `/ (root)`.
2. Open `https://<your-username>.github.io/<repo-name>/`.
3. To install it, use the **Install app** button (Android, Windows, Mac) or **Share > Add to Home Screen** (iPhone).

## Run the Streamlit apps

```
pip install -r requirements.txt
streamlit run app.py            # Advisor
streamlit run paper_trader.py   # Paper Trader
```

To host them free, use Streamlit Community Cloud (share.streamlit.io) and pick the repo and the main file.

## Notes

- The web app reads Yahoo Finance prices through free public proxies, which can be slow or busy at times.
- Signals are rule-based (trend, RSI, MACD, volume, strength vs Nifty) and backtested on past prices. Past results never guarantee future returns.
