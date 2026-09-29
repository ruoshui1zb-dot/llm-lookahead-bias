"""
Step 3: 让 DeepSeek 做四类判断（temperature = 0）
  recall : 只给公司名、代码和日期，不给任何文本，问股价相对大盘涨还是跌（记忆探测）
  guard  : 同上，但明确要求「只能使用该日期之前的信息」
  orig   : 给新闻稿原文开头，预测涨跌
  anon   : 给匿名化后的新闻稿，预测涨跌
"""
import os, re, json, concurrent.futures as cf
import pandas as pd
from importlib.machinery import SourceFileLoader
b = SourceFileLoader("b", "02_build.py").load_module()

ev = pd.read_csv("events.csv", dtype={"cik": str})
NAME = {t: v[0] for t, v in b.ALIAS.items()}

def fmt_date(d):
    d = pd.Timestamp(d); return f"{d.strftime('%B')} {d.day}, {d.year}"

PRED_SYS = ("You are an equity analyst. Read the excerpt from a company's quarterly earnings press release. "
            "Predict whether the company's stock will outperform or underperform the S&P 500 index from the close "
            "before the release to the close of the next trading day. Answer with one word: up or down.")
ONE_WORD = "Answer with one word: up or down."

def recall_q(t, d):
    return (f"{NAME[t]} ({t}) released its quarterly earnings results on {fmt_date(d)}. From the close before the "
            f"release to the close of the next trading day, did its stock outperform (up) or underperform (down) the S&P 500?")

def guard_q(t, d):
    return (f"Today is {fmt_date(d)}. {NAME[t]} ({t}) is releasing its quarterly earnings results today. You must not use "
            f"any information from after today. From the close before the release to the close of the next trading day, "
            f"will its stock outperform (up) or underperform (down) the S&P 500?")

TASKS = {
    "recall": lambda r: (ONE_WORD, recall_q(r.ticker, r.date)),
    "guard":  lambda r: (ONE_WORD, guard_q(r.ticker, r.date)),
    "orig":   lambda r: (PRED_SYS, r.orig),
    "anon":   lambda r: (PRED_SYS, r.anon),
}

def parse(a):
    a = a.lower()
    m = re.search(r"\b(up|down)\b", a)
    return m.group(1) if m else "NA"

if __name__ == "__main__":
    for task, build in TASKS.items():
        cache = f"pred_{task}.json"
        done = json.load(open(cache)) if os.path.exists(cache) else {}
        todo = [r for r in ev.itertuples() if r.accession not in done]
        print(task, "todo", len(todo), flush=True)
        with cf.ThreadPoolExecutor(32) as ex:
            futs = {ex.submit(b.ds, *build(r), 5): r.accession for r in todo}
            for f in cf.as_completed(futs):
                ans, mdl = f.result(); done[futs[f]] = [ans, mdl]
        json.dump(done, open(cache, "w"))
        ev[task] = ev.accession.map(lambda a: parse(done[a][0]))
        print(task, ev[task].value_counts().to_dict(), flush=True)
    ev.to_csv("events_pred.csv", index=False)
