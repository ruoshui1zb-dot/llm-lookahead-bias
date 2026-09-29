"""
Step 1: 收集业绩公告事件
- 公司：78 家美国大公司
- 事件：SEC EDGAR 上 2022-01 至 2026-09 的 8-K（Item 2.02 经营业绩），取随附新闻稿（通常为 EX-99.1）
- 股价：Yahoo Finance 复权收盘价，另取 SPY 作为大盘
"""
import os, json, re, time, html, urllib.request, concurrent.futures as cf
import pandas as pd

UA = {"User-Agent": "Lizhongbin research lizhongbin0827@ruc.edu.cn"}
TICKERS = """AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO ORCL CRM ADBE AMD INTC CSCO IBM QCOM TXN MU NFLX PYPL
JPM BAC WFC C GS MS AXP V MA BLK SCHW JNJ PFE MRK ABBV LLY UNH CVS AMGN GILD BMY MDT TMO ABT
WMT COST HD LOW TGT NKE SBUX MCD KO PEP PG CL MDLZ DIS CMCSA T VZ XOM CVX COP BA CAT DE GE HON
LMT RTX UPS FDX MMM F GM UBER ABNB""".split()
START, END = "2022-01-01", "2026-09-20"
PR_PAT = re.compile(r"(?i)ex[-_]?99[-_.]?0?1(?!\d)|ex991|dex991|press|release|earnings|pr\.htm|pr[-_]")
NOT_PR = re.compile(r"(?i)99[-_.]?0?[2-9](?!\d)")

def get(url, tries=7):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return r.read()
        except Exception:
            time.sleep(min(2 ** k, 40))       # SEC 限流时指数退避
    return None

def clean_text(raw):
    s = raw.decode("utf-8", "ignore")
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = html.unescape(s).replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()

def pick_doc(items, primary):
    docs = [i for i in items if i["name"].lower().endswith((".htm", ".html")) and i["name"] != primary
            and not re.match(r"(?i)^(R\d+\.htm|FilingSummary)", i["name"]) and "index" not in i["name"].lower()]
    pat = [i for i in docs if PR_PAT.search(i["name"]) and not NOT_PR.search(i["name"])]
    if pat:
        return pat[0]["name"]
    rest = [i for i in docs if not NOT_PR.search(i["name"])]
    return max(rest, key=lambda i: int(i.get("size") or 0))["name"] if rest else None

tk = json.loads(get("https://www.sec.gov/files/company_tickers.json"))
CIK = {v["ticker"]: (str(v["cik_str"]).zfill(10), v["title"]) for v in tk.values()}

def one_firm(t):
    os.makedirs("firm_cache", exist_ok=True)
    fp = f"firm_cache/{t}.json"
    events = json.load(open(fp)) if os.path.exists(fp) else []
    have = {e["accession"] for e in events}
    c, name = CIK[t]
    raw_sub = get(f"https://data.sec.gov/submissions/CIK{c}.json")
    if raw_sub is None:
        print(t, "submissions failed", flush=True); return events
    sub = json.loads(raw_sub)
    rec = pd.DataFrame(sub["filings"]["recent"])
    for f in sub["filings"].get("files", []):          # 超过 1,000 份的更早文件在附加 JSON 中
        if f.get("filingTo", "9999") >= START:
            extra = get(f"https://data.sec.gov/submissions/{f['name']}")
            if extra:
                rec = pd.concat([rec, pd.DataFrame(json.loads(extra))])
    rec = rec[(rec.form == "8-K") & rec["items"].str.contains("2.02") & (rec.filingDate >= START) & (rec.filingDate <= END)]
    if len(have) >= len(rec):                          # 已经齐全
        return events
    for r in rec.itertuples():
        if r.accessionNumber in have:
            continue
        acc = r.accessionNumber.replace("-", "")
        base = f"https://www.sec.gov/Archives/edgar/data/{int(c)}/{acc}"
        idx = get(f"{base}/index.json"); time.sleep(0.3)
        if not idx:
            continue
        doc = pick_doc(json.loads(idx)["directory"]["item"], r.primaryDocument)
        if not doc:
            continue
        raw = get(f"{base}/{doc}"); time.sleep(0.3)
        if not raw:
            continue
        events.append(dict(ticker=t, cik=c, sec_name=name, date=r.filingDate, accept=r.acceptanceDateTime,
                           accession=r.accessionNumber, doc=doc, text=clean_text(raw)[:4000]))
    json.dump(events, open(fp, "w"))
    print(t, len(rec), len(events), flush=True)
    return events

def prices(t):
    p1 = int(pd.Timestamp("2021-12-01").timestamp()); p2 = int(pd.Timestamp("2026-09-29").timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t}?period1={p1}&period2={p2}&interval=1d&events=div,split"
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as r:
                d = json.loads(r.read())["chart"]["result"][0]
            ts = pd.to_datetime(d["timestamp"], unit="s").normalize()
            return pd.Series(d["indicators"]["adjclose"][0]["adjclose"], index=ts, name=t).dropna()
        except Exception:
            time.sleep(2 * (k + 1))
    raise RuntimeError(t)

if __name__ == "__main__":
    events = []
    with cf.ThreadPoolExecutor(6) as ex:
        for r in ex.map(one_firm, TICKERS):
            events += r
    pd.DataFrame(events).to_csv("events_raw.csv", index=False)
    px = pd.concat([prices(t) for t in TICKERS + ["SPY"]], axis=1)
    px.index = px.index.date
    px.to_csv("prices.csv")
    print("events:", len(events), " price days:", len(px))
