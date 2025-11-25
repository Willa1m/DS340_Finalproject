# DS340 Final Project

## 项目概述
基于市场价格与按日对齐的新闻情感进行 EUR/USD 次日预测。新闻来源包含 Yahoo Finance RSS、InvestingLive（ForexLive）与 ECB，支持一周分析与多来源模型对比，并在训练中加入验证集与早停以减少过拟合。

## 环境与依赖
- Python `3.13`
- 必要包：`feedparser`、`nltk`、`vaderSentiment`、`yfinance`、`numpy`、`scikit-learn`、`torch`、`requests`、`beautifulsoup4`、`pandas`、`python-dateutil`

安装依赖：
```
python -m pip install feedparser nltk vaderSentiment yfinance numpy scikit-learn torch requests beautifulsoup4 pandas python-dateutil
```

## 运行
- 终端输出版（打印一周价格、新闻、训练日志与预测）：
```
python news_prediction.py
```

- Markdown 报告版（保留全部功能，生成 `output.md`）：
```
python generate_markdown.py
```
生成内容包含：
- 一周 EUR/USD 价格（由 `USDEUR=X` 反转得到）
- 三来源新闻条目与平均情感（代码块展示）
- 每来源训练日志（代码块展示）
- 次日 EUR/USD 预测与情感摘要

## 文件结构
- `news_prediction.py`：主流程，按日情感注入，多来源对比，验证集与早停，汇率方向修正
- `generate_markdown.py`：从主流程复用核心能力，捕获输出并生成结构化 Markdown 报告 `output.md`
- `withDataset.py`：原始基线脚本
- `.gitignore`：常见 Python/IDE/缓存文件忽略规则

## 说明与注意
- 网络请求包含基本错误处理与超时；若来源页面结构变化，可能需要调整解析选择器。
- 一周区间默认为 `2025-11-18` 至 `2025-11-25`，可在 `news_prediction.py` 中修改常量或扩展为命令行参数。
- 如果需要更强复现性，可添加 `requirements.txt` 并使用 `python -m pip install -r requirements.txt`。

## 验证
- 本地验证：确保运行成功并生成终端输出或 `output.md`
```
python news_prediction.py
python generate_markdown.py
```
- 远程验证（示例，Windows PowerShell）：
```
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/Willa1m/DS340_Finalproject/main/README.md"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/Willa1m/DS340_Finalproject/main/news_prediction.py"
```