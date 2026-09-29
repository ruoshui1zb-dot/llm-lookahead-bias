"""Step 1b: 下载复权收盘价（Yahoo Finance），含 SPY"""
import json, time, urllib.request
import pandas as pd
from importlib.machinery import SourceFileLoader

TICKERS = """AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO ORCL CRM ADBE AMD INTC CSCO IBM QCOM TXN MU NFLX PYPL
JPM BAC WFC C GS MS AXP V MA BLK SCHW JNJ PFE MRK ABBV LLY UNH CVS AMGN GILD BMY MDT TMO ABT
WMT COST HD LOW TGT NKE SBUX MCD KO PEP PG CL MDLZ DIS CMCSA T VZ XOM CVX COP BA CAT DE GE HON
LMT RTX UPS FDX MMM F GM UBER ABNB SPY""".split()

def prices(t):
    import os
    os.makedirs("price_cache", exist_ok=True)
    fp = f"price_cache/{t}.csv"
    if os.path.exists(fp):
        s = pd.read_csv(fp, index_col=0, parse_dates=True).iloc[:, 0]; s.name = t; return s
    p1 = int(pd.Timestamp("2021-12-01").timestamp()); p2 = int(pd.Timestamp("2026-09-29").timestamp())
    for host in ["query1", "query2"]:
        url = f"https://{host}.finance.yahoo.com/v8/finance/chart/{t}?period1={p1}&period2={p2}&interval=1d&events=div,split"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as r:
                d = json.loads(r.read())["chart"]["result"][0]
            ts = pd.to_datetime(d["timestamp"], unit="s").normalize()
            s = pd.Series(d["indicators"]["adjclose"][0]["adjclose"], index=ts, name=t).dropna()
            s.to_csv(fp); return s
        except Exception:
            time.sleep(10)
    # 备用：Nasdaq 历史行情（收盘价已做拆股调整，未做分红调整；两日窗口内影响很小）
    url = (f"https://api.nasdaq.com/api/quote/{t}/historical?assetclass={'etf' if t == 'SPY' else 'stocks'}"
           f"&fromdate=2021-12-01&todate=2026-09-29&limit=9999")
    for k in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh)", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                rows = json.loads(r.read())["data"]["tradesTable"]["rows"]
            s = pd.Series({pd.Timestamp(x["date"]): float(x["close"].replace("$", "").replace(",", "")) for x in rows},
                          name=t).sort_index()
            s.to_csv(fp); return s
        except Exception:
            time.sleep(10)
    raise RuntimeError(t)

if __name__ == "__main__":
    out = []
    for t in TICKERS:
        out.append(prices(t)); time.sleep(1)
    px = pd.concat(out, axis=1); px.index = px.index.date
    px.to_csv("prices.csv")
    print(px.shape, px.index.min(), px.index.max(), "max NA:", int(px.isna().sum().max()))
