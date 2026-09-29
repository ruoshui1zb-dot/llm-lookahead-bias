"""
Step 4: 分析与作图
- 记忆探测（recall）准确率随时间的变化 -> 推断「有效」训练截止日
- 2×2：原文 / 匿名 × 截止日前 / 后 的预测准确率与多空超额收益
- 推断：按公司聚类的 bootstrap 置信区间
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont("/Library/Fonts/Arial Unicode.ttf")
plt.rcParams.update({"font.family": "Arial Unicode MS", "axes.unicode_minus": False, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
import time
TS = time.strftime("%Y%m%d%H%M")
PFX = "Lizhongbin-LAB"          # 图片命名：作者-编号-FigNN-时间戳

SELF_CUTOFF = pd.Timestamp("2025-05-31")          # DeepSeek 自报的知识截止月份：2025-05
PRE_END, POST_START = pd.Timestamp("2025-01-31"), pd.Timestamp("2025-09-01")   # 截止日前后各留 3–4 个月灰色地带

ev = pd.read_csv("events_pred.csv", dtype={"cik": str}, parse_dates=["date"])
for k in ["recall", "guard", "orig", "anon"]:
    ev[f"hit_{k}"] = np.where(ev[k].isin(["up", "down"]), (ev[k] == "up").astype(int) == ev.up, np.nan)
    ev[f"ls_{k}"] = np.where(ev[k] == "up", ev.car, np.where(ev[k] == "down", -ev.car, np.nan))  # 按预测方向做多/做空
ev["period"] = np.select([ev.date <= PRE_END, ev.date >= POST_START], ["截止日前", "截止日后"], "灰色地带")
ev["half"] = ev.date.dt.year.astype(str) + np.where(ev.date.dt.month <= 6, "H1", "H2")
ev["big"] = ev.car.abs() >= ev.car.abs().median()

def boot(df, col, B=2000, seed=1):
    """按公司聚类的 bootstrap：返回均值与 95% 区间"""
    rng = np.random.default_rng(seed)
    g = {t: d[col].dropna().values for t, d in df.groupby("ticker")}
    keys = list(g)
    est = np.nanmean(np.concatenate(list(g.values())))
    draws = []
    for _ in range(B):
        pick = rng.choice(keys, len(keys))
        draws.append(np.mean(np.concatenate([g[k] for k in pick])))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return est, lo, hi

def boot_diff(df, a, b, B=2000, seed=2):
    rng = np.random.default_rng(seed)
    g = {t: d[[a, b]].dropna() for t, d in df.groupby("ticker")}
    keys = list(g)
    full = pd.concat(g.values()); est = (full[a] - full[b]).mean()
    draws = [pd.concat([g[k] for k in rng.choice(keys, len(keys))]).pipe(lambda x: (x[a] - x[b]).mean()) for _ in range(B)]
    return est, *np.percentile(draws, [2.5, 97.5])

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print("N events:", len(ev), "firms:", ev.ticker.nunique(), " up share:", ev.up.mean().round(3))
    print(ev.period.value_counts().to_dict())
    print("model tags:", "see pred json")

    # ---------- 1. 记忆探测随时间 ----------
    rows = []
    for h, d in ev.groupby("half"):
        for k in ["recall", "guard"]:
            e, lo, hi = boot(d, f"hit_{k}", B=500)
            rows.append([h, k, len(d), e, lo, hi])
        e, lo, hi = boot(d[d.big], "hit_recall", B=500); rows.append([h, "recall_big", d.big.sum(), e, lo, hi])
    rc = pd.DataFrame(rows, columns=["half", "task", "n", "acc", "lo", "hi"])
    print(rc.round(3).to_string(index=False))

    # ---------- 2. 2×2 ----------
    rows = []
    for p in ["截止日前", "截止日后"]:
        d = ev[ev.period == p]
        for k in ["recall", "guard", "orig", "anon"]:
            a = boot(d, f"hit_{k}"); l = boot(d, f"ls_{k}")
            rows.append([p, k, d[f"hit_{k}"].notna().sum(), *a, *[x * 1e4 for x in l]])
    tab = pd.DataFrame(rows, columns=["period", "task", "n", "acc", "acc_lo", "acc_hi", "ls_bp", "ls_lo", "ls_hi"])
    print(tab.round(3).to_string(index=False))
    for p in ["截止日前", "截止日后"]:
        d = ev[ev.period == p]
        print(p, "orig-anon acc diff:", np.round(boot_diff(d, "hit_orig", "hit_anon"), 3),
              " recall-guard:", np.round(boot_diff(d, "hit_recall", "hit_guard"), 3))
    # DiD：截止日前 (原文-匿名) 减 截止日后 (原文-匿名)
    rng = np.random.default_rng(3)
    def dd(df):
        pre, post = df[df.period == "截止日前"], df[df.period == "截止日后"]
        return (pre.hit_orig - pre.hit_anon).mean() - (post.hit_orig - post.hit_anon).mean()
    keys = ev.ticker.unique(); g = {t: d for t, d in ev.groupby("ticker")}
    draws = [dd(pd.concat([g[k] for k in rng.choice(keys, len(keys))])) for _ in range(1000)]
    print("DiD:", round(dd(ev), 3), np.percentile(draws, [2.5, 97.5]).round(3))
    tab.to_csv("tab_2x2.csv", index=False); rc.to_csv("tab_recall_time.csv", index=False)

    # ---------- 图 1：时间轴示意 ----------
    fig, ax = plt.subplots(figsize=(10, 2.9))
    ax.set_xlim(2021.8, 2026.95); ax.set_ylim(-1.3, 1.6); ax.axis("off")
    ax.annotate("", xy=(2026.9, 0), xytext=(2021.85, 0), arrowprops=dict(arrowstyle="->", lw=1.4))
    for y in range(2022, 2027):
        ax.plot([y, y], [-0.06, 0.06], color="k", lw=1); ax.text(y, -0.25, str(y), ha="center", fontsize=10)
    ax.axvspan(2022.0, 2025.08, ymin=0.47, ymax=0.62, color="#E6B0AA")
    ax.text(2023.5, 0.27, "截止日前样本：模型训练时可能见过公告和股价反应", ha="center", fontsize=10)
    ax.axvspan(2025.67, 2026.72, ymin=0.47, ymax=0.62, color="#AED6F1")
    ax.text(2026.2, 0.27, "截止日后样本", ha="center", fontsize=10)
    c = 2025.0 + 5 / 12
    ax.plot([c, c], [-0.4, 1.35], color="#C0392B", ls="--", lw=1.5)
    ax.text(c, 1.42, "模型自报训练截止：2025-05", ha="center", color="#C0392B", fontsize=10)
    ax.plot([2026.74, 2026.74], [-0.4, 1.0], color="#2E86C1", lw=1.5)
    ax.text(2026.74, 1.07, "调用日：2026-09", ha="center", color="#2E86C1", fontsize=10)
    ax.text(2023.5, -0.85, "回测「预测」的时间点 t：模型被假定站在 t 时刻，只知道 t 之前的信息", ha="center", fontsize=10, color="#555")
    fig.tight_layout(); fig.savefig(f"{PFX}-Fig01-{TS}.png", dpi=200)

    # ---------- 图 2：记忆探测准确率随时间 ----------
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    halves = sorted(ev.half.unique()); x = np.arange(len(halves))
    for k, col, lab, off in [("recall", "#C0392B", "只给公司和日期（记忆探测）", -0.12),
                             ("guard", "#E59866", "同上，并要求「不得使用该日之后的信息」", 0.0),
                             ("recall_big", "#6C3483", "记忆探测：只看反应幅度较大的一半事件", 0.12)]:
        s = rc[rc.task == k].set_index("half").reindex(halves)
        ax.errorbar(x + off, s.acc, yerr=[s.acc - s.lo, s.hi - s.acc], fmt="o-", color=col, capsize=3, ms=5, label=lab)
    ax.axhline(0.5, color="#888", lw=1, ls=":")
    cx = halves.index("2025H1") + 0.4 if "2025H1" in halves else None
    if cx is not None:
        ax.axvline(cx, color="#C0392B", ls="--", lw=1); ax.text(cx + 0.05, ax.get_ylim()[1] * 0.98, "自报截止 2025-05", color="#C0392B", fontsize=9, va="top")
    ax.set_xticks(x); ax.set_xticklabels(halves); ax.set_ylabel("判断正确的比例")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{PFX}-Fig02-{TS}.png", dpi=200)

    # ---------- 图 3：2×2 ----------
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
    for j, (col, lo, hi, yl) in enumerate([("acc", "acc_lo", "acc_hi", "判断正确的比例"), ("ls_bp", "ls_lo", "ls_hi", "按预测方向多空的平均超额收益 (bp)")]):
        for i, (k, c, lab) in enumerate([("orig", "#C0392B", "新闻稿原文"), ("anon", "#2E86C1", "匿名化文本")]):
            s = tab[tab.task == k].set_index("period").reindex(["截止日前", "截止日后"])
            xx = np.arange(2) + (i - 0.5) * 0.36
            ax[j].bar(xx, s[col], 0.36, color=c, label=lab, yerr=[s[col] - s[lo], s[hi] - s[col]], capsize=3)
        ax[j].set_xticks(range(2)); ax[j].set_xticklabels(["截止日前", "截止日后"]); ax[j].set_ylabel(yl)
        if j == 0: ax[j].axhline(0.5, color="#888", lw=1, ls=":"); ax[j].set_ylim(0.3, None)
        else: ax[j].axhline(0, color="k", lw=0.8)
    ax[0].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{PFX}-Fig03-{TS}.png", dpi=200)
