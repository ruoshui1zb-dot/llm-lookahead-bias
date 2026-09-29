# LLM 的前视偏差：DeepSeek 业绩公告实验

配套推文：《大模型做预测，会不会提前看过答案？》（李忠彬，连享会）

## 文件

| 文件 | 内容 |
|---|---|
| `code/01_collect.py` | 从 SEC EDGAR 下载 78 家美国大公司 2022-01 至 2026-09 的 8-K（Item 2.02）业绩新闻稿 |
| `code/01b_prices.py` | 下载日收盘价（Yahoo Finance，失败时用 Nasdaq 备用），含 SPY |
| `code/02_build.py` | 计算两日超额收益，截取新闻稿开头，调用 DeepSeek 匿名化并用正则兜底 |
| `code/03_predict.py` | 四类判断：记忆探测、约束提示、原文预测、匿名预测 |
| `code/04_analysis.py` | 记忆探测时间序列、2×2 比较、按公司聚类的 bootstrap，生成 3 张图 |
| `output/events_pred.csv` | 1,223 个事件：日期、超额收益、新闻稿摘录、匿名文本、四类判断 |
| `output/pred_*.json` `output/anon_cache.json` | DeepSeek 原始输出 |
| `output/tab_*.csv` `output/analysis_output.txt` | 分析结果 |
| `output/probes.txt` | 正文第 1 节的定性记忆探测问答 |
| `figs/` | 推文中的 3 张图 |

## 运行

```bash
cd code
pip install -r requirements.txt
export DEEPSEEK_API_KEY=你的key
python 01_collect.py      # SEC 有访问频率限制，约需 1 小时，可断点续跑
python 01b_prices.py
python 02_build.py
python 03_predict.py
python 04_analysis.py
```

股价数据来自 Yahoo Finance / Nasdaq，按其使用条款不在仓库中分发，运行脚本自动下载。

## 实验设定

- 模型：DeepSeek API `deepseek-chat`（API 返回 `deepseek-flash`），`temperature = 0`
- 调用日期：2026-09-29 至 2026-09-30
- 截止日前样本：公告日 ≤ 2025-01-31（818 个事件）；截止日后样本：公告日 ≥ 2025-09-01（262 个事件）
- 结果变量：公告日前一交易日收盘至公告日后一交易日收盘，个股收益减 SPY 收益
