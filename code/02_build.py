"""
Step 2: 构造结果变量、截取新闻稿开头、匿名化
- 结果：公告日前一交易日收盘 -> 公告日后一交易日收盘，个股收益减 SPY 收益；up = 超额收益 > 0
- 文本：新闻稿正文前约 1,200 个字符
- 匿名化：先让 DeepSeek 按规则改写，再用正则兜底检查公司名、股票代码和年份
"""
import os, re, json, time, urllib.request, concurrent.futures as cf
import numpy as np, pandas as pd

ev = pd.read_csv("events_raw.csv", dtype={"cik": str})
px = pd.read_csv("prices.csv", index_col=0, parse_dates=True)

# ---------- 超额收益 ----------
days = px.index
def window_ret(t, d):
    d = pd.Timestamp(d)
    i = days.searchsorted(d)                  # 公告日（或其后第一个交易日）
    if i < 1 or i + 1 >= len(days) or t not in px:
        return np.nan
    p0, p1 = px[t].iloc[i - 1], px[t].iloc[i + 1]
    s0, s1 = px["SPY"].iloc[i - 1], px["SPY"].iloc[i + 1]
    return (p1 / p0 - 1) - (s1 / s0 - 1)
ev["car"] = [window_ret(t, d) for t, d in zip(ev.ticker, ev.date)]
ev = ev.dropna(subset=["car"]).copy()
ev["up"] = (ev.car > 0).astype(int)

# ---------- 截取正文 ----------
def excerpt(s, n=1200):
    s = str(s)
    # 去掉 EDGAR 文件头，例如「EX-99.1 2 xxx.htm EX-99.1 Document Exhibit 99.1」
    m = re.search(r"(?i)exhibit\s*99\.?1\b", s[:300])
    if m:
        s = s[m.end():]
    else:
        s = re.sub(r"^(EX-99\.?\d*\s+\d+\s+\S+\s+)?(EX-99\.?\d*\s+)?(Document\s+)?", "", s)
    s = s.strip(" -–—:")
    cut = s[:n]
    return cut[: cut.rfind(".") + 1] if "." in cut[600:] else cut
ev["orig"] = ev.text.map(excerpt)
# 只保留真正的业绩新闻稿：需包含财务结果类词语（剔除 Tesla 交付量公告等）
FIN = r"(?i)revenue|net sales|net income|earnings per share|per diluted share|EPS|operating income|net revenues"
ev = ev[(ev.orig.str.len() > 300) & ev.orig.str.contains(FIN)].copy()

# ---------- 公司名称词表（用于兜底替换与检查） ----------
ALIAS = {
    "AAPL": ["Apple", "iPhone", "Mac", "iPad"], "MSFT": ["Microsoft", "Azure", "Windows", "Xbox"],
    "NVDA": ["NVIDIA", "Nvidia", "GeForce"], "AMZN": ["Amazon", "AWS"], "GOOGL": ["Alphabet", "Google", "YouTube"],
    "META": ["Meta", "Facebook", "Instagram", "WhatsApp"], "TSLA": ["Tesla"], "AVGO": ["Broadcom"],
    "ORCL": ["Oracle"], "CRM": ["Salesforce"], "ADBE": ["Adobe"], "AMD": ["AMD", "Advanced Micro Devices"],
    "INTC": ["Intel"], "CSCO": ["Cisco"], "IBM": ["IBM"], "QCOM": ["Qualcomm", "QUALCOMM", "Snapdragon"],
    "TXN": ["Texas Instruments", "TI"], "MU": ["Micron"], "NFLX": ["Netflix"], "PYPL": ["PayPal", "Venmo"],
    "JPM": ["JPMorgan", "J.P. Morgan", "Chase"], "BAC": ["Bank of America", "BofA", "Merrill"],
    "WFC": ["Wells Fargo"], "C": ["Citigroup", "Citi"], "GS": ["Goldman Sachs", "Goldman"],
    "MS": ["Morgan Stanley"], "AXP": ["American Express", "Amex"], "V": ["Visa"], "MA": ["Mastercard"],
    "BLK": ["BlackRock", "iShares"], "SCHW": ["Schwab"], "JNJ": ["Johnson & Johnson", "J&J"],
    "PFE": ["Pfizer", "Comirnaty", "Paxlovid"], "MRK": ["Merck", "Keytruda"], "ABBV": ["AbbVie", "Humira", "Skyrizi", "Rinvoq"],
    "LLY": ["Lilly", "Mounjaro", "Zepbound"], "UNH": ["UnitedHealth", "Optum", "UnitedHealthcare"], "CVS": ["CVS", "Aetna"],
    "AMGN": ["Amgen"], "GILD": ["Gilead"], "BMY": ["Bristol Myers Squibb", "Bristol-Myers Squibb", "BMS"],
    "MDT": ["Medtronic"], "TMO": ["Thermo Fisher"], "ABT": ["Abbott"], "WMT": ["Walmart", "Sam's Club"],
    "COST": ["Costco"], "HD": ["Home Depot"], "LOW": ["Lowe's", "Lowe’s"], "TGT": ["Target"], "NKE": ["NIKE", "Nike", "Jordan", "Converse"],
    "SBUX": ["Starbucks"], "MCD": ["McDonald's", "McDonald’s"], "KO": ["Coca-Cola", "Coke"], "PEP": ["PepsiCo", "Pepsi", "Frito-Lay"],
    "PG": ["Procter & Gamble", "P&G"], "CL": ["Colgate-Palmolive", "Colgate"], "MDLZ": ["Mondelez"],
    "DIS": ["Disney", "ESPN", "Hulu"], "CMCSA": ["Comcast", "NBCUniversal", "Peacock", "Xfinity"], "T": ["AT&T"],
    "VZ": ["Verizon"], "XOM": ["ExxonMobil", "Exxon Mobil", "Exxon"], "CVX": ["Chevron"], "COP": ["ConocoPhillips"],
    "BA": ["Boeing"], "CAT": ["Caterpillar"], "DE": ["Deere", "John Deere"], "GE": ["GE Aerospace", "General Electric", "GE"],
    "HON": ["Honeywell"], "LMT": ["Lockheed Martin", "Lockheed"], "RTX": ["RTX", "Raytheon", "Pratt & Whitney", "Collins Aerospace"],
    "UPS": ["UPS"], "FDX": ["FedEx"], "MMM": ["3M"], "F": ["Ford"], "GM": ["General Motors", "GM", "Chevrolet", "Cadillac"],
    "UBER": ["Uber"], "ABNB": ["Airbnb"],
}
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December|Jan\\.|Feb\\.|Mar\\.|Apr\\.|Aug\\.|Sept\\.|Sep\\.|Oct\\.|Nov\\.|Dec\\."

def regex_scrub(s, t):
    for a in sorted(ALIAS.get(t, []), key=len, reverse=True):
        s = re.sub(r"(?<![A-Za-z])" + re.escape(a) + r"(?![a-z])", "the Company", s)
    if len(t) >= 2:
        s = re.sub(r"\b" + re.escape(t) + r"\b", "XXX", s)
    s = re.sub(r"\((?:NYSE|NASDAQ|Nasdaq|NasdaqGS)\s*:\s*[A-Z.]+\)", "", s)
    s = re.sub(r"(?:" + MONTHS + r")\s+\d{1,2},?\s+(?:19|20)\d{2}", "[date]", s)
    s = re.sub(r"(?:" + MONTHS + r")\s+\d{1,2}", "[date]", s)
    s = re.sub(r"\b(?:19|20)\d{2}\b", "[year]", s)
    return s

def leaks(s, t):
    bad = [a for a in ALIAS.get(t, []) if len(a) > 2 and re.search(r"(?<![A-Za-z])" + re.escape(a) + r"(?![a-z])", s)]
    if re.search(r"\b(?:19|20)\d{2}\b", s):
        bad.append("year")
    return bad

KEY = os.environ.get("DEEPSEEK_API_KEY", "")
ANON_PROMPT = (
    "Rewrite the following excerpt from a company's earnings press release so that the company, the time period "
    "and the people cannot be identified. Rules: (1) replace the company name and any brand, subsidiary or product "
    "names with 'the Company' or a generic description such as 'a flagship product'; (2) replace people's names with "
    "their role, e.g. 'the CEO'; (3) remove ticker symbols and stock exchange names; (4) replace calendar years, "
    "specific dates and fiscal-year labels with relative terms such as 'the current quarter', 'the prior-year quarter' "
    "or 'the fiscal year'; (5) replace city names with 'City'. Keep every number (revenue, earnings per share, growth "
    "rates, margins, guidance) and all other wording unchanged. Output only the rewritten text.")

def ds(system, user, max_tokens=800, retries=5):
    body = json.dumps({"model": "deepseek-chat", "temperature": 0, "max_tokens": max_tokens,
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
    for k in range(retries):
        try:
            req = urllib.request.Request("https://api.deepseek.com/chat/completions", data=body,
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            return out["choices"][0]["message"]["content"].strip(), out.get("model", "")
        except Exception:
            time.sleep(2 ** k)
    return "API_ERR", ""

if __name__ == "__main__":
    cache = "anon_cache.json"
    done = json.load(open(cache)) if os.path.exists(cache) else {}
    todo = [(a, s) for a, s in zip(ev.accession, ev.orig) if a not in done]
    print("anonymize todo:", len(todo), flush=True)
    with cf.ThreadPoolExecutor(24) as ex:
        futs = {ex.submit(ds, ANON_PROMPT, s): a for a, s in todo}
        for j, f in enumerate(cf.as_completed(futs)):
            done[futs[f]] = f.result()[0]
            if (j + 1) % 200 == 0:
                json.dump(done, open(cache, "w")); print(j + 1, flush=True)
    json.dump(done, open(cache, "w"))
    ev["anon_llm"] = ev.accession.map(done)
    ev["anon"] = [regex_scrub(a, t) for a, t in zip(ev.anon_llm, ev.ticker)]
    ev["leak_before_scrub"] = [len(leaks(a, t)) > 0 for a, t in zip(ev.anon_llm, ev.ticker)]
    ev["leak_after_scrub"] = [len(leaks(a, t)) > 0 for a, t in zip(ev.anon, ev.ticker)]
    ev.drop(columns=["text"]).to_csv("events.csv", index=False)
    print("events:", len(ev), " up share:", ev.up.mean().round(3),
          " leak before/after scrub:", ev.leak_before_scrub.mean().round(3), ev.leak_after_scrub.mean().round(3))
