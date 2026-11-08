# us-market-radar 现有源码只读审计报告

> 报告文件：`intraday_scalper/AUDIT.md`
> 审计对象：仓库 `E:\wendang\deepseek-harness\default-workspace`（GitHub: `2827173579/us-market-radar`，分支 `main`）
> 审计时点：2026-11-06 18:27（Asia/Seoul），本地 HEAD = `193996d8`（`⚡ [实盘同步] 第 10 轮极速刷新`）
> 审计执行：Agent Teams 成员 `repo-auditor`（任务 `task-1`）；不属于被审计仓库既有代码

---

## 0. 审计范围与只读声明

### 0.1 审计范围

| 项目 | 内容 |
| :--- | :--- |
| 代码规模 | git 跟踪文件 63 个；主代码 `integrated_market_radar.py` 4,577 行、`ml_predictor_5y.py` 4,095 行、`v31_strategy.py` 1,050 行、`backtest_5y_engine.py` 2,153 行、`create_futu_terminal.py` 4,035 行 |
| 运行时 | 唯一工作流 `.github/workflows/market_radar.yml`（97 行，6 组 cron）；本地无独立后台服务 |
| 测试 | 21 个 Python 测试文件（静态统计 295 个 `test_*` 方法）+ 2 个 Node 前端测试；`test_e2e_report.json` 记录 55/55 |
| 数据缓存 | `data_cache/ml_panel_5y.parquet`、`data_cache/ml_extended_panel.parquet`、`data_cache/latest_ai_predictions.json`、4 个点开头 JSON 缓存 |
| 文档 | `README.md`、`TEST_INFRA.md`、`TEST_READY.md`、`V31_PRODUCTION_PLAN.md`、`backtest_report_5y.md`、`v31_quantitative_strategy_master_report.md` |

### 0.2 只读声明（实际执行动作清单）

- **未修改、未删除、未覆盖任何既有文件**；本报告是在 `intraday_scalper/AUDIT.md` 唯一新增的文件。
- **未执行任何写操作的 git 命令**：未 `commit`、未 `push`、未 `reset`、未 `fetch`、未 `checkout`、未 `pull`。只执行了 `git log`、`git status`、`git remote -v`、`git rev-parse`、`git ls-files`、`git show --stat`、`git rev-list` 等只读查询。
- **未运行** `integrated_market_radar.py`、`ml_predictor_5y.py`、`backtest_5y_engine.py` 以及任何测试套件（避免联网、耗时和副作用）。
- **未发起任何网络请求**（包括 Finnhub、yfinance、PushPlus、GitHub API）。所有 Finnhub/数据权限结论均标注为“源码证据”或“未验证”。
- **未输出、未读取任何密钥值**。仅确认：根目录无 `.env`（只有 `.env.example`，内容为占位符）；`.env` 被 `.gitignore:10` 忽略；密钥由 Actions secrets 注入（`market_radar.yml:83-85`）。`intraday_scalper/.env` 存在但未读取。
- parquet / JSON 仅用 pandas（`intraday_scalper/.venv` 的 Python 3.12.14 + pandas 3.0.1 + pyarrow 26.0.1）做**只读** `read_parquet` / `json.load` 检查，未写入任何数据文件。

### 0.3 方法与限制

- 大文件按 `grep` 命中行号分段 `read`，未逐行通读 4,577/4,095 行等超大文件；对入口函数、数据源、限速、风控、缓存、测试、CI 等关键路径做了定点取证。未覆盖的行可能存在本报告未提及的细节。
- 本机 `intraday_scalper/.venv` 未安装 `lightgbm`、`scikit-learn`、`yfinance`、`plotly`、`python-dotenv`；系统 Python 也无 `pyarrow`（只有本项目 venv 有）。因此**未实际训练 LightGBM、未实际运行 yfinance/回测**。
- 仓库根目录 **缺少** `.backtest_5y_cache.pkl`（被 `.gitignore:6` 的 `*.pkl` 忽略），因此 `v31_strategy.py` 的回测和 `test_e2e_v31.py` Tier 4 在干净 clone 中**无法直接运行**；本报告对其声称的 5 年业绩标注为“未验证/不可复现”。
- 本地 clone 可能落后远端：审计期间未 `git fetch`，HEAD 与本地 `origin/main` 引用相同（`193996d8`），但该仓库每约 90 秒产生远端提交（见 §1.3）。

### 0.4 关键结论速览（TL;DR）

1. **方案对两个主项目的形态描述大体符实**：`/quote` + 限速/429 退避、SOXL/SOXS/SMH/SOXX 标的、市场时段、暴跌熔断、缓存、5 年日线历史、LightGBM + 精确 23 维特征、未来 5 日三重障碍标签均能在源码中找到明确证据。
2. **但“现有数据能力”与日内策略需求严重不匹配**：现有代码**没有任何 Finnhub candle/WebSocket 调用**；唯一的“日内数据”是 `yfinance` 的 `period="2d", interval="5m", prepost=True`（`integrated_market_radar.py:665`）。`/quote` 只是快照（last/d/o/h/l/pc/t），无法提供逐笔成交或 1 分钟 bar。
3. **仓库 `main` 是一条数据流而非稳定代码分支**：Actions 机器人每约 90 秒 `git add/commit/pull --rebase/push` 一次，历史 5,678 个提交中 3,533 个来自 `actions@github.com`；本地 clone 每分钟都可能落后。新项目若与现有工作区/分支混用，会持续冲突。
4. **不存在 Firstrade 自动下单**：全仓库无券商 API、无委托下单代码；“Firstrade”仅出现在回测摩擦模型（`v31_strategy.py:59-61`）和推送给人的文字指令（`integrated_market_radar.py:2060-2074`）。
5. **“现有回测/测试”不能直接当新项目的可信基准**：`.backtest_5y_cache.pkl` 缺失；`backtest_report_5y.html/md` 所写 V3.1 Track 数字在当前代码中找不到生成路径；两套回测引擎滑点参数互相矛盾（10 bps vs 1.5 bps），文档之间数字也不一致（详见 §6.3/§6.4）。

---

## 1. 仓库与运行时全景

### 1.1 文件与项目关系

```
us-market-radar/（单仓库）
├─ integrated_market_radar.py      # 项目一：实时看盘雷达 + 本地 HTTP 服务 + 云端轮询守护 + 自动 git 同步（4,577 行）
├─ ml_predictor_5y.py              # 项目二：5 年日线面板 → 23 维特征 → LightGBM 双向概率（4,095 行，离线训练式工具）
├─ create_futu_terminal.py         # 被项目一 import：生成 futu_pro_terminal.html；并重绘 predict_radar_5y.html
├─ v31_strategy.py                 # V3.1 双轨策略回测引擎（1,050 行；依赖 .backtest_5y_cache.pkl）
├─ backtest_5y_engine.py           # 另一套独立 5 年回测引擎（2,153 行；自建 pickle 缓存与 HTML 战报）
├─ test_*.py（21 个）/ test_*.js（2 个）
├─ data_cache/*.parquet / *.json   # 训练面板与 AI 预测快照
├─ .options_cache.json / .earnings_cache.json / .radar_alert_history.json / .terminal_notifications.json
├─ *.html（index/futu_pro_terminal/macro_radar/stocks_radar/predict_radar_5y/backtest_report_5y）
└─ .github/workflows/market_radar.yml  # 唯一 CI：定时运行项目一，并自动 commit/push 产物
```

证据：`integrated_market_radar.py:36`（`from create_futu_terminal import build_futu_terminal_html`）；`create_futu_terminal.py:236-253`（重绘 AI 页面）；`v31_strategy.py:287-315`（读取 5 年 pkl）；`backtest_5y_engine.py:29`（`import yfinance`）；测试文件清单见 §6.1。

### 1.2 唯一 GitHub Actions 工作流

文件：`.github/workflows/market_radar.yml`（97 行）。

| 项 | 内容 | 证据 |
| :--- | :--- | :--- |
| 名称 | `US Market Radar 24-7 Engine` | `market_radar.yml:1` |
| 定时 1/2 | `cron: '0 0 * * 1-5'`（夜盘前半场）/`'0 4 * * 1-5'`（夜盘后半场），注释称每 2 分钟 | `market_radar.yml:5-8` |
| 定时 3/4/5/6 | `'0 8'`、`'0 12'`、`'0 16'`、`'30 20'`（均周一至周五） | `market_radar.yml:9-16` |
| 手动触发 | `workflow_dispatch`，可传 `duration_minutes`（默认 60）/`interval_seconds`（默认 60） | `market_radar.yml:17-28` |
| 并发 | `concurrency: group: us-market-radar, cancel-in-progress: true` | `market_radar.yml:30-32` |
| 权限 | `permissions: contents: write` | `market_radar.yml:34-35` |
| 运行器 | `ubuntu-latest`，Python 3.10，`actions/checkout@v4`（`fetch-depth: 10`） | `market_radar.yml:38-51` |
| 依赖 | `pip install -r requirements.txt`（仅 5 个包） | `market_radar.yml:53-56`；`requirements.txt:1-5` |
| 本地优先避让 | 若 `git log -n 10` 中发现非 `actions@github.com` 提交且距今 <900 秒，则 schedule 运行直接跳过 | `market_radar.yml:58-78` |
| 主命令 | `python integrated_market_radar.py --loop-daemon --duration-minutes $DURATION --interval-seconds $INTERVAL`；schedule 时 `DURATION` 默认 `240`、`INTERVAL` 默认 `60`，两个夜盘 cron 强制 120 | `market_radar.yml:80-97` |

**静态发现的调度事实（重要）**：
- schedule 触发的每个 job 都是 **240 分钟**（`market_radar.yml:89-92`；注释中“#5 270 分钟 / #6 210 分钟”与实际代码不一致）。
- 6 组 cron 首尾相接近乎覆盖周一至周五全天，但 **20:00–20:30 UTC 存在约 30 分钟空窗**（16:00 启动的 job 到 20:00 结束，20:30 才启动下一个）；`cancel-in-progress` 还会让 00:00 的新 job 取消上一轮 20:30 启动、本应运行到 00:30 的 job。
- **工作流只运行项目一**，不运行 `ml_predictor_5y.py`、不运行任何测试（`market_radar.yml:80-97` 是唯一执行步骤）。
- 仓库内**没有其他 workflow**（`git ls-files .github` 只有 1 个文件）。

### 1.3 自动 commit/push 行为（90 秒级）

`--loop-daemon` 模式每轮在扫描后执行（`integrated_market_radar.py:4488-4514`）：

1. 将 `futu_pro_terminal.html` 复制为 `index.html`（`:4491-4492`）；
2. 用 `GH_TOKEN`/`GITHUB_TOKEN` 把 remote URL 改写成带 token 的地址（`:4494-4498`）；
3. `git add` 固定 9 个产物文件（index/futu_pro_terminal/predict_radar_5y/`data_cache/latest_ai_predictions.json`/通知与缓存 JSON/macro/stocks HTML）（`:4499-4502`）；
4. `git commit -m "⚡ [实盘同步] 第 N 轮极速刷新 (时间)"`（`:4503-4506`）；
5. `git pull --rebase --autostash -X theirs origin main`（`:4507`，注意 `-X theirs` 会以远端内容解决冲突）；
6. `git push origin HEAD:main`（`:4508`）。

**实测 git 历史证据**（只读命令）：

| 事实 | 数值/证据 |
| :--- | :--- |
| 总提交数 | 5,678（`git rev-list --count HEAD`） |
| 作者分布 | `actions@github.com` 3,533 个；`a18669358277@gmail.com` 2,145 个（`git shortlog -sne`） |
| 时间跨度 | 2026-10-28 09:05 → 2026-11-06 09:23（`git log --reverse/--max-count`） |
| 提交节奏 | 连续 `[实盘同步]` 提交间隔恰为 **90 秒**：09:13:14 / 09:14:44 / 09:16:15 / 09:17:45 / 09:19:15 / 09:20:45 / 09:22:15 / 09:23:45 |
| 单日提交量 | 2026-11-05 约 965 个提交（`git log --format=%ad --date=short | Group-Object`） |
| HEAD 提交内容 | 每轮改动 7 个 HTML/JSON 产物，合计约 2,397 行增删（`git show --stat HEAD`） |

**对本地开发的直接影响**：
- 每约 90 秒远端 `main` 前进一次；本地 clone 在几分钟内即落后，`git status` 会显示工作区产物与 HEAD 差异。
- `git pull --rebase -X theirs` + 机器人 `git add` 固定文件列表，意味着**本地同时修改这些文件或在同一工作区提交，极易被 rebase/覆盖或制造冲突**。
- 本地提交（非 actions 作者）会让云端 job 触发“本地活跃避让”而跳过最多 15 分钟（`market_radar.yml:58-78`、`integrated_market_radar.py:4465-4479`）；反之云端每 90 秒的提交也会打断本地 `git pull --rebase` 的假设。
- 自动 push 使用 PAT/`GITHUB_TOKEN`，触发 GitHub Pages 部署（`:4494-4510`）；仓库实际上兼作**公开数据发布通道**（README:13-17 列出 Pages 地址）。

### 1.4 “现有后台进程”实况

| 假设形态 | 实际证据 | 结论 |
| :--- | :--- | :--- |
| 常驻后台服务/Windows 服务 | 审计时点 `Get-CimInstance Win32_Process` 无 python 进程；`Get-ScheduledTask` 无 radar/market/quant/stock 相关任务 | **本机不存在** |
| `--loop-daemon` 进程 | 只在 `--loop-daemon` 参数下存在，是 **Actions 作业内的 while 循环**（`integrated_market_radar.py:4443-4521`），job 结束即退出 | 云端作业内循环，不是常驻服务 |
| 本机无参默认模式 | 无参数运行时是**前台**无限循环：每 1 秒探测锁文件，满 60 秒执行一轮（`integrated_market_radar.py:4533-4568`；`SYNC_INTERVAL = 60` 见 `:478-480`），退出即停止 | 前台守护，不是后台进程 |
| 本地 `.bat` 启动器 | `create_desktop_shortcuts.py:37-45` 生成的是 `--once` 单次刷新脚本（且路径写死 `C:\Users\User\...`），非常驻进程 | 单次任务 |
| 本地服务端口 | `start_star_api_server(8765)` 在 `--once` 或轮询模式下额外拉起 HTTP 线程/子进程（`integrated_market_radar.py:433-476`） | 仅在运行脚本期间存在 |

因此，Gemini 方案中“现有后台进程”应修正为：**云端 GitHub Actions 定时作业 + 作业内循环；本地只有手动启动的前台守护**。本机当前时点没有任何雷达后台进程。

### 1.5 本地/远端状态、缺失文件与环境污染

- `git status --porcelain`：仅 `?? intraday_scalper/`（新项目目录未跟踪）；既有文件无改动。
- `.radar_last_sync` 不存在（`.gitignore:3` 忽略；锁文件由运行中的雷达写入，`integrated_market_radar.py:478-500`）。
- `.backtest_5y_cache.pkl` 不存在（`.gitignore:6` `*.pkl` 忽略）；`TEST_INFRA.md:12` 与 `v31_strategy.py:291-293` 都把它当作权威数据源 → **干净 clone 无法运行 5 年回测**。
- `data_cache/*.parquet` 与多个含账户持仓信息的 JSON 被 git 跟踪（`git ls-files`），仓库公开页面已含实盘持仓示例（`user_portfolio.json:1-38`）；新项目不得把密钥/账户信息写进被跟踪文件。
- 根 `.gitignore` **不忽略** `.venv/`、`intraday_scalper/`；新项目目录当前整体为未跟踪状态，若执行 `git add -A` 会把 `.venv`（数千文件）加入提交 —— 属于新项目自己的隔离风险（见 §8.4）。

---

## 2. 两个主项目与周边模块逐项证据表

### 2.1 `integrated_market_radar.py`（项目一）

| 维度 | 事实 | 证据（file:line） |
| :--- | :--- | :--- |
| 定位 | 实时看盘雷达 + Futu 风格终端 + 微信预警 + 本地 HTTP 服务 + 云端 24/7 轮询 | 文件头 `:1-12`；`main()` `:4400-4577` |
| 标的池 | 代码实际 **16 只 ETF + 55 只个股 = 71 只**（AST 只读解析 `MONITORED_TARGETS`）；文件头却写 “37 标的”“57 标的”，与代码不一致 | `:46-152`（头注释 `:3-5`，字典 `:49-152`） |
| ETF 关键标的 | SMH、SOXX、SOXL、SQQQ、SOXS、VIXY、QQQM、SPY、KORU、IBIT 等 | `:51-66`（SMH `:55`、SOXX `:56`、SOXL `:57`、SQQQ `:64`、SOXS `:65`） |
| Finnhub 报价 | `GET /api/v1/quote`，字段 c/d/dp/h/l/o/pc/t | `:562-592`（URL `:564`，字段 `:571-581`） |
| Finnhub 财报日历 | `GET /api/v1/calendar/earnings`，4 天分片、12h 本地缓存 | `:756-804`（URL `:785`，TTL 43200s `:770`，写缓存 `:801-802`） |
| 历史日线 | yfinance 批量 `period="1y"` + 单标的回退 `interval="1d"` | `:2423-2460`（`:2428`、`:2444`） |
| 日内 5 分钟 | yfinance `period="2d", interval="5m", prepost=True`，算 VWAP/5m RSI/RVOL/盘前盘后价 | `:647-751`（`:665`，指标 `:694-745`） |
| 期权数据 | yfinance `Ticker.option_chain()`，30 分钟缓存、6 线程并发、71 只标的 | `:842-1085`（`:854-857` 缓存、`:864` 6 线程、`:875`/`:903` 取链、`:1080` 写缓存） |
| 微信推送 | PushPlus `https://www.pushplus.plus/send`，双通道（clawbot + wechat） | `:1586-1633`（URL `:1598`，token `:1595`，双通道 `:1601-1605`） |
| 本地缓存/状态 | `.options_cache.json`、`.earnings_cache.json`、`.radar_alert_history.json`、`.terminal_notifications.json`、`target_watchlist.json`、`user_portfolio.json` | `:198-204`、`:753-754` |
| 输出产物 | `macro_radar.html`、`stocks_radar.html`、`futu_pro_terminal.html`（另有 index 复制） | `:198-200`；`create_futu_terminal.py:1-5` |
| 模式 | `--once` 单次 / `--loop-daemon` 云端循环 / 无参前台 60s 循环 / `--star-server-daemon`（8765 端口） / `--test-wechat` | `:4400-4456`、`:4443-4521`、`:4523-4577` |
| 自动 git 同步 | 仅 `--loop-daemon` 分支：add/commit/pull/push | `:4488-4514` |

### 2.2 `ml_predictor_5y.py`（项目二）

| 维度 | 事实 | 证据（file:line） |
| :--- | :--- | :--- |
| 定位 | Qlib 风格长表面板 → 因果特征 → 三重障碍打标 → Walk-Forward 训练 → HTML/JSON 战报 | 文件头 `:1-12`；`main()` `:3998-4091` |
| 标的池 | **28 只目标股（7 赛道）+ 7 只宏观 ETF = 35** | `:84-134`（`SECTOR_MAP` `:84-120`、`MACRO_ETFS` `:132`、`ALL_SYMBOLS` `:134`） |
| 宏观 ETF | `['QQQM','SPY','SMH','IGV','TLT','VIXY','GLD']`（**不含 SOXL/SOXS/SOXX**） | `:132` |
| 特征 | **精确 23 维** = 15 股票技术面 + 2 跨资产 + 6 宏观，含 `assert len(FEATURE_COLS) == 23` | `:136-170`（定义 `:137-167`，断言 `:170`） |
| 特征因果性 | 无 `shift(-k)`/无 bfill，左连接宏观特征，禁止宽表全局 dropna | `:325-498`（文档 `:325-328`）、`:500-562`（`:511` 注释） |
| 历史数据 | 默认起点 `2021-11-01`，yfinance `auto_adjust=True`；缓存 `data_cache/ml_panel_5y.parquet` | `:214-257`（`:214`、`:218`）、`:260-318`（缓存读 `:277`、写 `:314`） |
| 三重障碍标签 | T+1 开盘入场（`Open.shift(-1)`）；多空双向；动态 ATR 止盈止损；T+5 超时结算；最后 5 日 Label=NaN 隔离 | `:586-764`（入场 `:615-616`；eff_atr `:569-583`；多空触轨 `:685-748`；尾部 `:655`/`:794`） |
| LightGBM | `lgb.LGBMClassifier`，max_depth=5 / num_leaves=23 / lr=0.03 / balanced 等；多空两个独立模型 | `:68-78`（import）、`:904-930`（live refit）、`:964-1016`（walk-forward 参数/训练） |
| 交叉验证 | Expanding Window，测试年 2024/2025/2026，**5 日 purge**，含因果断言 | `:771-819`（`:773` test_years、`:774` purge_days、`:817` 断言） |
| 实时报价 | 可选 `--realtime`：Finnhub `/quote` 覆盖最新日线 | `:1202-1266`（URL `:1223`）、`:1269+`（覆盖）；`main` `:4013-4016` |
| 扩展标的 | `EXTENDED_RADAR_SECTOR_MAP` 覆盖 43 个扩展 symbol（含 SOXX/SOXL/SOXS），yfinance 抓取 2023 起并缓存 | `:3711-3763`；`:3766-3863`（`:3797` 起始 2023-01-01、`:3807` 写 parquet） |
| 输出 | `predict_radar_5y.html` + `data_cache/latest_ai_predictions.json`（keys: kpi/macro_weather/table1..4/top10/top_long/top_short/all_predictions/by_ticker/sectors） | `:3866-3940`（导出）、`render_html_dashboard` `:2435`；JSON 只读检查见 §5.2 |
| 模型持久化 | **无**：全仓库无 `joblib`/`pickle`/`save_model`/`lgb.Booster.save_model` 写入；每次运行重新训练 | grep 全仓库无命中；训练后仅返回内存对象 `:923-930` |

### 2.3 `v31_strategy.py`（V3.1 双轨回测引擎）

| 维度 | 事实 | 证据 |
| :--- | :--- | :--- |
| 配置 | Track1/Track2 工厂；滑点 `0.00015`（1.5 bps）、佣金 0、最大 4/3 持仓、25% 单股上限、1.5% 风险预算、-6.8% 硬止损 | `:52-120`（`:58-67`、`:88-120`） |
| 数据入口 | `load_indicators_cache()` 依次找 `.backtest_5y_cache.pkl`，三处路径均可能不存在，找不到则抛 `FileNotFoundError` | `:287-315`（`:290-294`、`:300-301`） |
| 执行/仓位 | T+1 开盘执行；卖单优先；跳空过滤；结构止损；风险平价仓位；$200 最小下单 | `:466-513`（跳空 `:476`、止损 `:480-482`、风险平价 `:492-497`、最小单 `:510`） |
| 依赖 | **不 import 另外两个主项目**；由 `test_v31_sandbox.py`、`test_e2e_v31.py` 调用 | `test_v31_sandbox.py:34-37`、`test_e2e_v31.py:52-57` |

### 2.4 `backtest_5y_engine.py`（第二套独立回测引擎）

| 维度 | 事实 | 证据 |
| :--- | :--- | :--- |
| 标的 | `CORE_STOCKS` 42 只 + `MACRO_BENCHMARKS = ['QQQ','SMH']`；**不含 SOXL/SOXS/SOXX** | `:38-44` |
| 滑点 | `SLIPPAGE_RATE = 0.001`（**0.1% = 10 bps**），与 README/计划/报告宣称的 1.5 bps 不同 | `:46` |
| 数据 | yfinance `period="5y"` 多线程抓取，写 `.backtest_5y_cache.pkl`（pickle，被 gitignore） | `:481-529`（`:505`、`:526-528`） |
| 模式 | `NEXT_OPEN`（T 日信号 → T+1 开盘成交）与 `SAME_CLOSE`（理论对照）两种运行 | `:2138-2144` |
| 产物 | 写 `backtest_report_5y.html`（模板内为 “V5.2 机构双录对账版” 风格） | `:2121-2123`、`:2131` |
| 与 V3.1 的关系 | **是独立第二套引擎**，不引用 `v31_strategy.py`；两套引擎指标口径/滑点不同 | 全文件无 v31 import；`:46` vs `v31_strategy.py:60` |

### 2.5 `create_futu_terminal.py` / 前端静态资产

| 维度 | 事实 | 证据 |
| :--- | :--- | :--- |
| 定位 | 生成 `futu_pro_terminal.html`；“富途”只是 UI 风格，**无富途 OpenD/券商接口** | 文件头 `:1-5`；全仓库 grep 无 `OpenD/futu_api`（仅 HTML/文案） |
| AI 预测注入 | 读取已有 `data_cache/latest_ai_predictions.json`，用本轮 Finnhub/盘前盘后价格覆盖 `latest_price/atr_pct/target/stop`，并用启发式规则重算 `predicted_win_rate_pct`/`calibrated_win_rate_pct` | `:61-69`（读 JSON）、`:84-147`（覆盖与重算）、`:149-198`（排序/动作） |
| 时间戳重标 | 每轮把 `kpi.sync_time` 改为当前时间、`data_mode` 改为“Finnhub 秒级实时/ yfinance 最新收盘价” | `:205-207` |
| AI 页面重绘 | 读取 `ml_predictor_5y.render_html_dashboard` 重绘 `predict_radar_5y.html`（**不重新训练模型**） | `:236-253` |
| 页面自刷新 | `<meta http-equiv="refresh" content="90">` | `:272` |

> **注意**：`predict_radar_5y.html` 的“AI 胜率”会被上述规则根据实时价/宏观状态二次调整（`:120-147`），而底层模型概率来自上一次完整 `ml_predictor_5y.py` 训练（模型不落盘）。因此看板上的“AI 实时预测”是**旧模型概率 + 新价格/规则修正**的混合结果，不是每轮重新训练的模型输出。

### 2.6 测试套件

静态统计（AST，仅解析不执行）：21 个 `test_*.py`，合计 **295 个 `test_*` 方法**；另有 2 个 Node 测试（`test_frontend_stress.js`、`test_frontend_adversarial.js`）。最大套件为 `test_e2e_v31.py`（55 个测试方法）与 `test_v32_circuit_breaker.py`（17 个）。详见 §6。

---

## 3. 数据源 / API / 限速 / 退避 / 频率

### 3.1 Finnhub（现有代码只用了 2 个 endpoint）

| 用途 | endpoint | 位置 | 频率/缓存 |
| :--- | :--- | :--- | :--- |
| 实时快照报价 | `GET /api/v1/quote?symbol=...&token=...` | `integrated_market_radar.py:564`；`ml_predictor_5y.py:1223` | 项目一每轮最多 60 只；项目二仅 `--realtime` 时 35 只 |
| 财报日历 | `GET /api/v1/calendar/earnings?from=...&to=...` | `integrated_market_radar.py:785` | 4 天分片 × 最多 12 次/轮，12h 缓存（`:770`） |
| **candle / WebSocket** | **未使用** | 全仓库 grep `stock/candle`、`ws.finnhub`、`websocket` 在既有跟踪文件中 **0 命中**（匹配仅出现在本次新增的 `intraday_scalper/`，不属于被审计代码） | — |

`/quote` 返回字段映射（last/change/pct_change/high/low/open/prev_close/timestamp）：`integrated_market_radar.py:571-581`、`ml_predictor_5y.py:1233-1242`。它是**最新成交快照**，不是历史 bar、也不是逐笔流。

### 3.2 限速与 429 退避（两个项目实现不一致）

**项目一：`FinnhubRateLimiter`（`integrated_market_radar.py:502-560`）**

| 参数/行为 | 数值 | 证据 |
| :--- | :--- | :--- |
| 免费额度红线（注释） | 60 次/分钟 | `:505-506` |
| 令牌桶容量/填充 | 50 次/分钟（0.833 req/s） | `:511-514`、`:560` |
| 强制最小间隔 | 1.05 秒 | `:515`、`:540-547` |
| 并发 | `max_workers=2` | `:594-600` |
| 429 退避 | 第 1 次 10 秒、第 2 次 20 秒，清空令牌桶；只重试 1 次，仍失败则该标的回退 yfinance | `:549-558`（10/20s）、`:582-587`（2 次尝试后回退） |
| 财报分片 | 同样走限速器 | `:787` |

**项目二：`fetch_finnhub_realtime_quotes`（`ml_predictor_5y.py:1202-1266`）**

| 参数/行为 | 数值 | 证据 |
| :--- | :--- | :--- |
| 限速 | **无令牌桶**，仅 `time.sleep(random.uniform(0.30,0.40))` + `max_workers=2` | `:1204`、`:1218`、`:1222`、`:1254` |
| 实际突发速率 | 2 并发 × 每 0.35s 一次 ≈ 5.7 req/s（约 340 次/分钟），**远超注释所称免费 60 次/分钟** | 由 `:1222`+`:1254` 推算 |
| 429 退避 | 仅 `sleep(2.0)` 一次，无指数退避 | `:1244-1249` |

> 结论：Gemini 方案“已有请求限速和 429 退避”对项目一**成立**；对项目二**只是有节流和轻微 429 处理，不满足免费档 60/min 的稳态约束**。新项目若复用 Finnhub，必须使用或重写项目一的令牌桶，不能复用项目二的实现。

### 3.3 yfinance（现有“历史/日内/期权”数据的主要来源）

| 用途 | 调用 | 证据 |
| :--- | :--- | :--- |
| 1 年日线 | `yf.download(tickers, period="1y", group_by="ticker", threads=True)` + 单标的 `yf.Ticker(t).history(period="1y", interval="1d")` | `integrated_market_radar.py:2423-2449` |
| 日内 5 分钟（含盘前盘后） | `yf.download(tickers, period="2d", interval="5m", prepost=True)` | `integrated_market_radar.py:665` |
| 期权链 | `yf.Ticker(t).option_chain(exp)`，用于 PCR/预期波动，30 分钟缓存、6 线程 | `integrated_market_radar.py:875-903`、`:857`、`:864` |
| 5 年日线（项目二） | `yf.download(ticker, start='2021-11-01', auto_adjust=True)` | `ml_predictor_5y.py:214-218` |
| 5 年日线（第二套回测） | `yf.download(tk, period="5y", auto_adjust=True)` | `backtest_5y_engine.py:505` |
| 扩展标的（2023 起） | `_fetch_single_ticker(tk, start_date="2023-01-01")` | `ml_predictor_5y.py:3797` |

**没有任何 1 分钟级历史数据源**：现有最细粒度是 yfinance 的 2 天 × 5 分钟（仅近 2 个交易日，且依赖 Yahoo 免费接口）；`/quote` 只是快照。Gemini 方案中“SOXL/SOXS 实时 1 分钟成交聚合 VWAP”所需的数据在现有代码中**不存在**。

### 3.4 PushPlus 通知

- 通过 `https://www.pushplus.plus/send` POST markdown 卡片（`integrated_market_radar.py:1598-1633`）。
- 双通道：配置的 `PUSHPLUS_CHANNEL`（默认 `clawbot`）+ 官方 `wechat` 兜底（`:1601-1605`）；成功判定 `code == 200`（`:1624`）。
- Token 只从环境变量读取（`:187-188`），`.env.example:4-5` 为占位符。
- Actions 通过 secrets 注入 `FINNHUB_API_KEY`、`PUSHPLUS_TOKEN`（`market_radar.yml:83-85`）。

### 3.5 其他数据源与执行接口

| 类别 | 状态 | 证据 |
| :--- | :--- | :--- |
| 富途/牛牛行情、OpenD、futu API | **不存在**（“富途”只是 UI 命名） | `create_futu_terminal.py:1-5`；全仓库 grep `OpenD/futu_api` 无命中 |
| Firstrade 交易 API / 自动下单 | **不存在**；只有回测摩擦校准与文字指令 | 回测配置 `v31_strategy.py:59-61`；推送文案指令 `integrated_market_radar.py:2060-2074`；SOP 人工下单 `V31_PRODUCTION_PLAN.md:156-163`、`:256-259` |
| 其他行情供应商（Polygon/Alpaca/IB 等） | 未发现 | 全仓库无相关 import/URL |

### 3.6 频率/时限/调度一览

| 机制 | 频率/时限 | 证据 |
| :--- | :--- | :--- |
| Actions schedule | 6 组 cron；每次 job 240 分钟（`workflow_dispatch` 可覆盖） | `market_radar.yml:5-16`、`:89-96` |
| 云端轮询间隔 | 默认 60s；两个夜盘 cron 强制 120s；下限 `max(30, ...)` | `market_radar.yml:91-96`；`integrated_market_radar.py:4454` |
| 本地无参轮询 | 60s（`SYNC_INTERVAL`） | `integrated_market_radar.py:478-480`、`:4556-4562` |
| 每轮耗时下限 | 60 只 quote × 1.05s ≈ 63s + yfinance/HTML/期权开销 → 实测提交间隔约 90s | `:540-547`、`:3905`；§1.3 的 git 时间戳 |
| 期权缓存 | 30 分钟（1800s） | `:857` |
| 财报缓存 | 12 小时（43200s）+ 同日校验 | `:770` |
| 微信去重冷却 | 同状态 2 小时；自选建仓 1 小时 | `:205-206`、`:1641` |
| 前端自动刷新 | 90 秒 meta refresh | `create_futu_terminal.py:272` |
| GitHub Pages 部署 | 每轮 push 触发 | `integrated_market_radar.py:4494-4510` |

---

## 4. 风险控制与状态机

### 4.1 市场时段状态机

`get_us_market_session()`（`integrated_market_radar.py:612-645`）基于 `America/New_York` 的 **weekday + 小时** 返回 5 种状态：

| 状态 | 判定 | 证据 |
| :--- | :--- | :--- |
| WEEKEND | 周五 ≥20:00、周六全天、周日 <20:00 | `:634-635` |
| OVERNIGHT | 周日 ≥20:00 及其余夜间 | `:636-637`、`:644-645` |
| REGULAR | 周一至周五 09:30–16:00 | `:638-639` |
| PRE_MARKET | 04:00–09:30 | `:640-641` |
| POST_MARKET | 16:00–20:00 | `:642-643` |

**缺口**：**没有美股节假日日历、没有提前收盘（13:00 半日市）处理**；仅用星期与小时近似。对日内策略而言，节假日/半日市的时段判断需要新实现。

### 4.2 暴跌熔断与 3 日冷却锁（V3.2）

**日线指标层**（`compute_indicators`，`integrated_market_radar.py:2462-2551`）：
- 暴跌定义：单日收益 ≤ **-6.0%** 或跳空低开 ≤ **-4.5%**（`:2529-2531`）。
- 派生状态：`Crash_5d`、`Crash_3d`、`Prev_Crash_3d`（`:2534-2536`），`Is_Right_Turn` 右侧反转 K 线（`:2538-2542`），暴跌日实体/收盘配对（`:2544-2549`）。

**日内建仓熔断**（`evaluate_intraday_entry`，`:1276-1563`）：

| 规则 | 公式/条件 | 证据 |
| :--- | :--- | :--- |
| ATR 双轨自适应崩跌阈值 | `crash_thresh = -max(3.8, min(5.2, 2.0*atr_pct))` | `:1350-1351` |
| 基础崩跌 | `pct_change <= crash_thresh` | `:1354` |
| 盘前/盘后强势反转解锁 | `safe_ext >= +3.0% 且 (pct+safe_ext) > -2.5%` 时取消基础崩跌 | `:1355-1357` |
| 断头铡刀（冲高跳水） | 距日内高点 ≤ -6.0% 且当日 ≤ -1.0% | `:1359-1361` |
| 综合 `is_crash_halt` | 基础崩跌 / 断头铡刀 / 盘后 ≤ -4.5% / (日内+盘后) ≤ -5.5% / 跌破 Fib 下沿 8% 且弱势 | `:1363-1369` |
| 熔断输出 | 不挂单、`is_crash_halt=True`、提示“暴跌宣泄期” | `:1371-1390` |
| 3 日冷却锁 | `prev_crash_3d or crash_3d` 时冻结；解锁 A（前日右侧反转阳线 + 今日 > -3.8% + 盘后 > -4.5%）或 B（日内 ≥ +2.5% 且收复大阴线实体 50% 且站上 VWAP/EMA5） | `:1395-1421`（`:1398`、`:1400-1401`） |
| 反向杠杆隔离 | `SQQQ/SOXS/VIXY` 走独立对冲分支，不参与黄金坑多头抄底 | `:1311-1345` |
| 对冲工具清仓/收割 | SMH RSI ≤33 或单日 ≥+7% 或偏离 EMA5 ≥6% → 逢高收割；跌破 EMA5 → 清仓 | `:2775-2803` |

### 4.3 仓位、止损、止盈

| 机制 | 公式/参数 | 证据 |
| :--- | :--- | :--- |
| 账户口径 | 固定参考本金 $10,000；单股上限 25%（$2,500）；1.5% 风险平价一次性建仓 | `integrated_market_radar.py:193-196` |
| 结构止损 | `max(min(sig_low-0.5*ATR, last-1.8*ATR), last*0.932)`（硬上限 -6.8%） | `integrated_market_radar.py:2755`；回测同式 `v31_strategy.py:480-481` |
| 风险平价仓位 | `target_notional = min(equity*25%, equity*1.5%/stop_dist)`，`stop_dist` 下限 3.5% | `integrated_market_radar.py:2756-2758`；`v31_strategy.py:493-497` |
| 观察标破位 | Fib 下沿 ×0.93 / ATR 偏离 ≤ -2.8 为极限破位；浅破需放量才止损 | `integrated_market_radar.py:2822-2843` |
| 持仓标止损 | 自定义防守价 / 亏损 ≤ -7% / 成本 - 2×ATR | `:2813-2820` |
| 冲顶清仓 | RSI ≥82，或 RSI ≥76 且 Bias≥20% 且滞涨且浮盈≥10% | `:2860-2866` |
| EMA 移动止盈 | 疯牛档（Bias5d≥12 或 RSI5d≥72 或浮盈≥12%）跌破 EMA10；常规档跌破 EMA20（Bias≥5 且近日高位） | `:2870-2912` |
| 强阻减仓 | 水下单反抽阻力减 50% | `:2849-2857` |
| 回测同款参数 | 4 槽位/25%/1.5%/-6.8%/$200 最小单/跳空过滤（开盘 < 信号低点或 < 昨收-1.5%，或 > 昨收+2%） | `v31_strategy.py:63-71`、`:476` |

**重要边界**：以上全部是**分析/提示/回测**级别风控，**没有任何自动平仓或自动委托**；熔断仅冻结“新开仓提示”，持仓卖出提示仍需人工执行。

### 4.4 宏观市场状态与一票否决

`evaluate_market_weather()`（`integrated_market_radar.py:2556-2631`）：
- VIXY 三因子（单日涨跌、MA20 偏离、60 日分位）、TLT 利率状态机、QQQM/SMH 相对强弱风格判断（`:2556-2562`）。
- 一票否决（Veto）：VIXY 恐慌脉冲 / QQQM 与 SMH 双破 MA20 / QQQM 急杀 / SMH 大跌 → 输出“严禁追高”文案（`:2612-2631`）。
- QQQ 闲置资金泊入建议：暴雨/破位时留现金，否则建议 MOO 泊入 QQQ（`:2675-2695`）。

### 4.5 预警去重状态机

- 状态定义：`NORMAL / STOP_LOSS / WATCHLIST_ENTRY_READY / GOLDEN_PIT / TAKE_PROFIT`（`integrated_market_radar.py:1694-1703`）。
- 冷却：同状态 2 小时、自选建仓 1 小时（`:205-206`、`:1641`）。
- 持久化：`.radar_alert_history.json`（`weather`/`targets`/`position_peaks`，`last_notified_at` 时间戳；只读检查见 §5.2）。
- 推送内容包含“Firstrade 9:30 开盘委托单”文字指令（`:2060-2074`），**不是机器下单**。

### 4.6 风控缺口（对日内新项目最关键）

1. 没有交易所节假日/半日市日历（`:612-645`）。
2. 没有自动清算/自动下单/券商对账（全仓库无券商 API）。
3. 没有逐笔/1 分钟数据，所有“日内”判断基于 5 分钟 Yahoo bar（`:665`）。
4. 熔断阈值是**标的自适应 3.8%~5.2%**，不是账户级回撤熔断；没有账户级 kill-switch。
5. 止损依赖人工在 Firstrade 挂 GTC（`V31_PRODUCTION_PLAN.md:161-163`），代码不管理订单状态。

---

## 5. 数据格式与缓存 schema

### 5.1 `data_cache/*.parquet`（pandas 只读检查）

| 文件 | shape | 列与 dtype | 时间范围 | 其他 |
| :--- | :--- | :--- | :--- | :--- |
| `data_cache/ml_panel_5y.parquet` | **(41,547, 7)** | `Date(datetime64) / Ticker(str) / Open,High,Low,Close(float64) / Volume(int64)` | **2021-11-01 → 2026-10-30** | 35 个 Ticker（=28 目标股+7 宏观 ETF）；总 NULL = 0 |
| `data_cache/ml_extended_panel.parquet` | **(30,539, 8)** | 上述 7 列 + `Sector(str)` | **2023-01-03 → 2026-11-04** | 36 个 Ticker（含 SOXX/SOXL/KORU/SKHY 等扩展），7 个 Sector；总 NULL = 0 |

要点：
- 项目二的 parquet **只缓存原始 OHLCV**，23 维特征与标签在运行时计算（`ml_predictor_5y.py:277` 读取、`:314` 写入、`:500-562` 构建特征）。
- `ml_panel_5y.parquet` 最新日期 **2026-10-30**，距审计日（2026-11-06）已 7 天；而 Actions 的 `git add` 列表**不包含任何 parquet**（`integrated_market_radar.py:4500`），因此该 ML 缓存**不会随云端刷新**，只能在本地手动 `--refresh` 更新 → 云端页面展示的 AI 概率可能基于过期面板（与 §2.5 的时间戳重标叠加时尤其要注意）。
- `ml_extended_panel.parquet` 含 2026-11-04 数据，来源是 `build_extended_live_features_df` 本地/手动运行（`ml_predictor_5y.py:3766-3807`）。

### 5.2 JSON 缓存 schema（只读 `json.load` 检查）

| 文件 | 顶层结构 | 关键字段 | 生产者/消费者 |
| :--- | :--- | :--- | :--- |
| `data_cache/latest_ai_predictions.json`（~299 KB） | dict，14 个顶层 key：`kpi / macro_weather / table1 / table2 / table3 / table1_short / table2_short / sectors / top10 / top_long / top_short / all_predictions / by_ticker / table4` | `all_predictions` 为 **77 条**记录，每条含 ticker/sector/最新价/ATR/bias/RSI/`model_raw_prob_pct`/`calibrated_win_rate_pct`/目标价/止损价/多空概率等 | 写：`ml_predictor_5y.py:3866-3940` 与 `create_futu_terminal.py:221-222`；读：`create_futu_terminal.py:66-69` |
| `.options_cache.json`（~50 KB） | dict：`schema_version(=3) / updated_at / data` | `data` 为 71 个 ticker；每项 21 字段：`has_options/options_exp/call_vol/put_vol/call_dollar/put_dollar/pcr/dollar_pcr/pcr_sentiment/top_call_info/call_wall/top_put_info/put_wall/iv_atm/exp_move_pct/…` | 写：`integrated_market_radar.py:1080`；TTL 1800s（`:857`） |
| `.earnings_cache.json`（~2 KB） | dict：`updated_at / base_date / data` | `data[ticker] = {date, hour(bmo/amc/"")}` | 写：`integrated_market_radar.py:801-802`；TTL 43200s（`:770`） |
| `.radar_alert_history.json`（~6 KB） | dict：`weather / targets / position_peaks` | `targets[ticker] = {state, last_notified_at}`（如 `STOP_LOSS`、`INTC_CRITICAL_STOP`、`NORMAL`） | 写/读：`integrated_market_radar.py:1568-1585` 等 |
| `.terminal_notifications.json`（~50 KB） | list，当前 88 条 | 每条：`id/time/category/cat_label/level/ticker/name/cat/price/pct_change/title/desc` | 读写：`integrated_market_radar.py:208-227` |
| `target_watchlist.json` | list，21 个 symbol | 自选列表（含 SOXL/SOXS/SOXX 等） | `integrated_market_radar.py:248-266` |
| `user_portfolio.json` | dict，3 个持仓 | `INTC / SOXL / SOXS` 的 shares/cost/trim/base_val/target/note（**含实盘持仓信息，已提交到公开仓库**） | `integrated_market_radar.py:228-246` |
| `test_e2e_report.json` | dict | `total_tests=55, passed=55, failed=0, duration≈9.6s, status=PASS`，时间戳 2026-11-03 | `test_e2e_v31.py:836-893` 生成 |

### 5.3 TTL 与新鲜度矩阵

| 数据 | TTL/刷新方式 | 是否随 Actions 更新 | 证据 |
| :--- | :--- | :--- | :--- |
| Finnhub /quote | 每轮（60/120s） | 是 | `:3902-3906` |
| yfinance 1y 日线 | 每轮重新下载 | 是 | `:3895-3896` |
| yfinance 2d 5m 日内 | 每轮重新下载 | 是 | `:3908-3910` |
| 财报日历 | 12h 缓存 | 是（会写 `.earnings_cache.json` 并提交） | `:770`、`:801-802`、`:4500` |
| 期权链 | 30min 缓存 | 是（`.options_cache.json` 在 add 列表） | `:857`、`:4500` |
| AI 预测 JSON | 只在完整 ML 运行时重写 | **否**（由 Actions 每轮“重标”而非重算） | `create_futu_terminal.py:61-69`、`:205-207`；`market_radar.yml:80-97` |
| ML parquet 面板 | 手动 `--refresh` | **否** | `ml_predictor_5y.py:274-287`；`integrated_market_radar.py:4500`（无 parquet） |
| 回测 pkl | 手动/回测引擎生成 | 否（gitignore） | `.gitignore:6`；`backtest_5y_engine.py:526-527` |

### 5.4 文档与代码/数据不一致清单（静态发现）

| 不一致点 | 文档说法 | 代码/数据实际 | 证据 |
| :--- | :--- | :--- | :--- |
| 标的数量 | README “70 只”；文件头 “37/57 只”；工作流步骤名 “65 只”；注释 “55 只（12 ETF+43 股）” | 代码 **16 ETF + 55 股 = 71**；AI JSON `all_predictions` 77 条 | `README.md:3`、`integrated_market_radar.py:3-5`、`market_radar.yml:80`、`integrated_market_radar.py:3903`、`:4464` 等 |
| V3.1 Track1 期末资产 | README：$132,374（+32.37%）；V31 计划/报告：$130,441.41（+30.44%）；胜率 README 56.8% vs 计划 58.3%；滑点 README $940 vs 计划 $658.22 | 两套文档互相矛盾，且无法从当前代码复现（见 §6.3） | `README.md:63-73`；`V31_PRODUCTION_PLAN.md:20-26`、`:33-42`；`backtest_report_5y.md:20-25`、`:36-48` |
| Track2 Calmar 验收线 | TEST_INFRA 验收标准 ≥ **2.00** | `test_e2e_v31.py` 实际断言 ≥ **0.70** | `TEST_INFRA.md:167`、`:189`；`test_e2e_v31.py:737-743` |
| 三重障碍 ATR 系数 | TEST_READY 文档：上轨 `max(0.035, 1.5*ATR%)`、下轨 `0.022` | 代码：上轨 `max(0.030, 1.35*eff_atr)`、下轨 `max(0.022, 1.00*eff_atr)`，且 `eff_atr` 另有软压缩 | `TEST_READY.md:76-78`；`ml_predictor_5y.py:665-666`、`:569-583` |
| 回测滑点 | README/计划/报告：Firstrade 1.5 bps | `backtest_5y_engine.py` 常量 **0.001 = 10 bps**（另一套引擎 v31 才是 1.5 bps） | `backtest_5y_engine.py:46`；`v31_strategy.py:60`；`backtest_report_5y.md:4` |
| 调度时长 | cron 注释 #5 “270 分钟”、#6 “210 分钟” | 代码统一 `DURATION=240`（schedule） | `market_radar.yml:13-16` vs `:89-92` |
| 锁文件/周期文案 | 多处打印 “180 秒 / 3 分钟轮询” | 实际 `SYNC_INTERVAL=60`、`--once` 提示 “180 秒” 不一致 | `integrated_market_radar.py:480`、`:4395`、`:4547`、`:4559`、`:4566` |
| 依赖声明 | `requirements.txt` 只有 5 个包 | 主代码还 import `lightgbm`、`sklearn`、`pyarrow`（parquet）、可选 `python-dotenv`；测试需 `pytest` | `requirements.txt:1-5`；`ml_predictor_5y.py:68-78`、`:277`、`:314`；`integrated_market_radar.py:159` |
| 文档引用缺失文件 | TEST_INFRA 引用 `ORIGINAL_REQUEST.md`、`PROJECT.md`、`DISPATCH.md`、`.backtest_5y_cache.pkl` | 这些文件均不存在于当前仓库 | `TEST_INFRA.md:10`、`:12`、`:186-192`；实际文件系统检查 |

---

## 6. 测试与回测基础设施

### 6.1 测试资产盘点（静态，未执行）

| 测试文件 | 类数 | `test_*` 方法数 | 依赖 |
| :--- | ---: | ---: | :--- |
| `test_e2e_v31.py` | 11 | **55** | `v31_strategy` + `.backtest_5y_cache.pkl`（Tier4） |
| `test_ml_predictor.py` | 15 | 27 | `ml_predictor_5y`（TEST_READY 记录 16/16 通过，1.3s） |
| `test_backtest_engine.py` | 1 | 23 | `backtest_5y_engine` |
| `test_challenger2_m1_empirical.py` | 3 | 18 | `integrated_market_radar` + **GitHub REST API** |
| `test_v32_circuit_breaker.py` | 1 | 17 | `integrated_market_radar` + `v31_strategy` |
| 其他 16 个文件 | — | 其余 | 多为 unittest 场景/对抗测试 |
| **合计** | — | **295** | 另有 2 个 Node 前端测试（`test_frontend_*.js`） |

证据：AST 静态统计；`test_e2e_report.json:1-11`（55/55 PASS，2026-11-03）；`TEST_READY.md:4-7`（16/16）；`TEST_INFRA.md:172-178`（e2e runner 说明）。

### 6.2 可运行性障碍（干净 clone 现状）

1. **硬编码作者机器路径**：
   - `test_e2e_v31.py:41-46`：`C:\Users\User\.gemini\antigravity-ide\scratch\us-market-radar`；
   - `test_v32_circuit_breaker.py:56`：同类路径；
   - `v31_strategy.py:291-293`：三次尝试中的额外路径；
   - `create_desktop_shortcuts.py:12-26`、`:41`：Windows 绝对路径与 `--once` 启动脚本。
   本机用户为 `28271`，这些路径不存在；从仓库根运行 `python test_*.py` 时因 sys.path[0] 仍是仓库根，部分 import 可侥幸成功，但没有任何路径是可移植的。
2. **缺少权威数据文件**：`.backtest_5y_cache.pkl` 不存在（`.gitignore:6`）；`test_e2e_v31.py` Tier4 `setUpClass` 直接 `load_indicators_cache()`（`:667-672`），`v31_strategy.py:300-301` 找不到即抛异常 → Tier4 会 error，而不是 skip。
3. **测试联网/依赖外部状态**：`test_challenger2_m1_empirical.py:237-271` 用 `GITHUB_TOKEN/GH_TOKEN` 调 GitHub Actions REST API，并硬编码断言某次 run `41544599045` 为 `cancelled` 且当前无 in-progress radar run；无 token 时 skip。这既依赖网络，也依赖历史 run 状态，属于**不可稳定复现的测试**。
4. **测试风格混杂**：`unittest` 为主，部分文件可被 pytest 收集；没有 pytest 配置/`conftest.py`。
5. **CI 不跑测试**：唯一 workflow 只运行项目一（`market_radar.yml:80-97`），没有任何 test step。

### 6.3 5 年回测报告可复现性核查（重点）

仓库里有 **两套 5 年回测实现**和 **两套互相矛盾的绩效数字**：

| 资产 | 实现 | 滑点 | 数据依赖 | 产物 |
| :--- | :--- | :--- | :--- | :--- |
| `backtest_5y_engine.py` | 独立引擎（42 只 CORE_STOCKS + QQQ/SMH；T+1 开盘/同收盘两模式） | **0.001（10 bps）** `:46` | 自建 `.backtest_5y_cache.pkl`（`:526-527`，gitignore） | `backtest_report_5y.html`（模板标题 “V5.2 机构双录对账版” `:2131`） |
| `v31_strategy.py` + `test_v31_sandbox.py` / `test_e2e_v31.py` | 双轨策略引擎（42 只 CORE_STOCKS + QQQ/SMH） | **0.00015（1.5 bps）** `v31_strategy.py:60` | 同一 `.backtest_5y_cache.pkl`（`v31_strategy.py:287-315`） | 仅控制台表格（`test_v31_sandbox.py:126-201`），**当前代码不写 HTML/MD** |

核查结论：
1. **`backtest_report_5y.html/md` 中的 V3.1 Track 数字无法由当前仓库代码一键生成**：
   - 全仓库 `grep "187,910|187910|130,441|130441|V3.1 Track 2"` 只在 `test_v31_sandbox.py`/`v31_strategy.py` 的显示代码中命中，**`backtest_5y_engine.py` 内没有任何 “187,910” 或 “V3.1 Track” 字符串**（grep 无命中）；其 HTML 模板标题为 “V5.2 机构双录对账版”。
   - `test_v31_sandbox.py` 只 `print` 表格，不写文件（`:204-207`）。
   - 因此 HTML/MD 报告更像**手工/半手工文档产物**，当前代码无法复现。
2. **两套引擎滑点不一致**（10 bps vs 1.5 bps），而报告宣称 1.5 bps（`backtest_report_5y.md:4`、`V31_PRODUCTION_PLAN.md:56`）。
3. **数据文件缺失**：`.backtest_5y_cache.pkl` 不在仓库（gitignore），两套引擎都需要先联网重建缓存（`backtest_5y_engine.py:500-528`；`v31_strategy.py:300-301`），而重建缓存使用的 yfinance 5 年数据与报告生成时的数据可能不同（复权/上市时间变化），**绩效数字不可稳定复现**。
4. `V31_PRODUCTION_PLAN.md` 与 `backtest_report_5y.md` 的 Track1/Track2 数字内部一致，但与 `README.md:63-73` 不一致（$130,441 vs $132,374；胜率/PF/滑点也不同）。

> 审计结论：**不得把现有 5 年回测结果当作新项目的可信 ground truth**；若需要，应先补回 `.backtest_5y_cache.pkl`、统一滑点口径、让报告由代码生成，再重跑验证。本审计未运行回测（无缓存、需联网）。

### 6.4 文档漂移汇总

见 §5.4 表；此外：
- `TEST_INFRA.md` 的验收标准（Calmar≥2.00、Track2 MDD≤-15%）与 `test_e2e_v31.py` 实际断言（Calmar≥0.70、Track2 MDD>-20%）不一致（`TEST_INFRA.md:159-168`、`:189` vs `test_e2e_v31.py:701-743`）。
- `TEST_READY.md` 宣称 16/16 通过（`:128-132`），是历史记录；当前环境未安装 LightGBM，无法复验。
- README 的 “55 项全量断言” 实际是 55 个 **unittest 方法**（`test_e2e_v31.py` 55 个 `test_*`），不是断言数；`test_e2e_report.json` 也只是作者机器的一次运行快照。

---

## 7. 「Gemini 方案 vs 实际源码」差异表

> 判定：**一致** = 源码明确支持；**部分一致** = 存在但范围/口径/实现与方案描述有差异；**不一致** = 源码与描述的形态/能力不符；**未验证** = 本审计无证据（未联网/未执行）。
> 所有条目均给出证据行号；没有证据的条目标注“未找到证据”。

| # | 方案假设（用户提供的 Gemini 方案要点） | 判定 | 证据（file:line） | 说明 |
| :-- | :--- | :--- | :--- | :--- |
| 1 | 项目一使用 Finnhub `/quote` | **一致** | `integrated_market_radar.py:564`、`:571-581`；`ml_predictor_5y.py:1223` | 两个项目都调用 `/quote`；项目一是主数据源，项目二仅 `--realtime` |
| 2 | 已有请求限速（rate limiter） | **部分一致** | 项目一：令牌桶 50/min + 1.05s 最小间隔 `:502-560`；项目二：无令牌桶，仅 0.30-0.40s 随机 sleep `ml_predictor_5y.py:1218-1254` | 项目一达标；项目二突发速率远高于免费 60/min，不能照搬 |
| 3 | 已有 429 退避（backoff） | **部分一致** | 项目一：10s/20s，仅 1 次重试后回退 yfinance `:549-558`、`:582-587`；项目二：仅 `sleep(2)` 一次 `ml_predictor_5y.py:1244-1249` | 项目一是“短退避 + 降级”，不是长指数退避；项目二更弱 |
| 4 | 已有 SOXL/SOXS/SMH/SOXX 等标的 | **部分一致** | 雷达/终端：`integrated_market_radar.py:55-57`（SMH/SOXX/SOXL）、`:65`（SOXS）、`:64`（SQQQ）；AI 扩展映射含 SOXX/SOXL/SOXS：`ml_predictor_5y.py:3718-3719`、`:3762`；但 **ML 核心 35 标的只有 SMH**：`:132`；**独立回测引擎只有 QQQ/SMH**：`backtest_5y_engine.py:44` | 只有“看盘/告警/扩展预测”覆盖杠杆半导体；训练与回测核心池不包含 SOXL/SOXS/SOXX |
| 5 | 已有市场时段判断 | **一致（但有缺口）** | `integrated_market_radar.py:612-645` | 5 状态（WEEKEND/OVERNIGHT/PRE/REGULAR/POST）；**无节假日/半日市日历** |
| 6 | 已有暴跌熔断 | **一致** | 指标层 `:2528-2549`；建仓熔断层 `:1350-1390`；3 日冷却 `:1395-1421`；断头铡刀 `:1359-1361`；反向 ETF 隔离 `:1311-1345` | 熔断只冻结“新开仓提示/挂单”，不自动平仓；是提示级风控 |
| 7 | 已有本地缓存 | **一致** | `.options_cache` 30min `:854-857`/`:1080`；`.earnings_cache` 12h `:753-757`/`:770`/`:801-802`；`.radar_alert_history`、`.terminal_notifications` `:201-204`；项目二 parquet 缓存 `ml_predictor_5y.py:277`/`:314` | 缓存格式见 §5 |
| 8 | 已有宏观市场状态 | **一致** | `evaluate_market_weather` `:2556-2631`（VIXY/TLT/QQQM/SMH/Veto）；QQQ 泊入建议 `:2675-2695`；ML 侧 `evaluate_macro_weather` `ml_predictor_5y.py:1304` | 宏观状态存在，但为日线级、规则式，不是实时 macro feed |
| 9 | 项目二使用五年历史日线数据 | **一致** | `ml_predictor_5y.py:214-218`（start=2021-11-01）、`:260-318`（parquet 缓存）；只读检查 `ml_panel_5y.parquet` 时间范围 **2021-11-01 → 2026-10-30**，35 标的 | 数据为 yfinance 日线（`auto_adjust=True`），非 Finnhub |
| 10 | 项目二使用 LightGBM | **一致** | `ml_predictor_5y.py:69`、`:904`、`:916`、`:1010`、`:1016`（`lgb.LGBMClassifier`） | 未安装 LightGBM 的环境会 `lgb=None` 并在训练处报错（`except ImportError` 只兜导入） |
| 11 | 项目二使用 23 维特征 | **一致（精确 23）** | `ml_predictor_5y.py:136-170`（15+2+6=23，`:170` 断言） | 特征定义与方案完全吻合 |
| 12 | 未来五个交易日三重障碍标签 | **一致** | `ml_predictor_5y.py:586-764`：T+1 `Open.shift(-1)` 入场 `:615-616`，多空双向 `:685-748`，T+5 超时结算 `:739-748`，尾部 5 日 NaN `:604`/`:794` | 进入价用的是“次日开盘”，不是当日收盘；方案表述未细化，此处以源码为准 |
| 13 | 项目二主要提供高周期方向与风险收益背景 | **一致（但看板被二次加工）** | 报告/预测生成 `ml_predictor_5y.py:3866-3940`、`:1560-2229`；但 `create_futu_terminal.py:84-147` 用实时价覆盖并重算胜率，`:205-207` 重标 `sync_time/data_mode` | 底层模型是日线级；看板上的“实时 AI 胜率”是旧模型 + 新价/规则修正，方案未提此加工 |
| 14 | 存在“现有后台进程” | **部分不一致** | 云端：Actions 作业内 `--loop-daemon` 循环 `integrated_market_radar.py:4443-4521`；本地：无参前台 60s 循环 `:4533-4568`、可选 8765 本地 HTTP 服务 `:433-476`；**审计时本机无 python 进程、无相关计划任务** | 没有独立常驻服务/系统服务；“后台”本质是 CI job + 前台守护 |
| 15 | 存在“现有 GitHub Actions 工作流” | **一致** | `.github/workflows/market_radar.yml:1-97`（唯一工作流，6 组 cron，240min 作业，每轮运行项目一） | 方案未提：job 内**自动 commit/push**、历史 3,533 个机器人提交、约 90s 一次（§1.3） |
| 16 | 存在“现有测试” | **一致（但不可直接跑）** | 21 个 py 测试文件、295 个 test 方法、2 个 JS 测试；`test_e2e_report.json` 55/55；`TEST_INFRA.md`/`TEST_READY.md` | 见 §6.2：硬编码路径、缺 `.backtest_5y_cache.pkl`、一个测试联网 GitHub API、CI 不跑测试 |
| 17 | 现有测试/工作流/后台进程“不得修改” | **无法从源码裁决（边界建议见 §8）** | 无证据表明有版本化护栏；实际风险：Actions 每 90s 覆盖 `main`、本地提交触发云端避让 | 这不是源码事实，而是交付约束；本审计只报告风险，不修改任何文件 |
| 18 | 存在 Firstrade 自动下单 | **不一致** | 全仓库无券商 API/下单函数；仅：回测摩擦模型 `v31_strategy.py:59-61`、文字推送指令 `integrated_market_radar.py:2060-2074`、人工 SOP `V31_PRODUCTION_PLAN.md:156-163`/`:256-259` | 实盘执行=人工在 Firstrade 下单；新项目若要求自动交易必须从零接入券商 |
| 19 | 免费 Finnhub 具备 candle / WebSocket 权限 | **未验证（源码中无使用）** | 既有跟踪文件 grep `stock/candle`、`ws.finnhub`、`websocket` **0 命中**；现有代码只依赖 `/quote` 与 `/calendar/earnings` | 权限实际状态需端点实测（本审计不联网）；但可确定：**现有代码从未使用 candle/WS，新项目无法“复用现有实时 K 线能力”** |
| 20 | 新项目需要“实时 1 分钟成交聚合 VWAP” | **不一致（现有数据不支持）** | 现有最细粒度：yfinance `2d/5m/prepost` `integrated_market_radar.py:665`；`/quote` 仅快照 `:571-581` | 无 1 分钟历史/逐笔；免费 Finnhub 是否可用需另测，不能假设已有 |

**与方案严重冲突/必须让用户知道的事实（补充）**：

1. **仓库 `main` 是每 90 秒自更新的数据发布分支**（3,533 个 Actions 提交；每轮 `git pull --rebase -X theirs` + `push`，`integrated_market_radar.py:4488-4514`）。新项目若放在同一工作区并提交，会与机器人提交流互相覆盖/冲突；本地提交还会让云端扫描暂停 15 分钟（`market_radar.yml:58-78`）。**必须隔离**（独立目录 + 独立分支/仓库 + 不触碰自动同步文件）。
2. **现有代码没有 candle/WS，也没有逐笔/1 分钟数据**：Gemini 方案若把“实时 1 分钟 VWAP 策略”建立在此数据能力上，属于**前提不成立**；必须先做数据权限实测与替代方案（付费 Finnhub、其他供应商或自行采集）。
3. **不存在 Firstrade 自动下单**：方案若假设“新项目不改现有实盘执行就可直接复用”，不成立；实盘执行是人工 SOP。
4. **现有 5 年回测数字不可复现**：`.backtest_5y_cache.pkl` 缺失、两套引擎滑点 10bps/1.5bps 矛盾、HTML/MD 中 V3.1 数字在当前代码中无生成路径；README 与 V31 计划的 Track1 数字也互相矛盾（$132,374 vs $130,441）。
5. **AI 预测“实时性”被包装**：`create_futu_terminal.py:84-147` 用实时价重算胜率/目标价、`:205-207` 把 `sync_time` 改成当前时间，而模型本身不落盘、不随 Actions 重训；ML parquet 最后日期 2026-10-30 且 Actions 不刷新 parquet（`integrated_market_radar.py:4500` 无 parquet）。新项目不能把该 JSON 当“新鲜模型输出”使用。
6. **依赖声明不完整**：`requirements.txt` 无 `lightgbm/scikit-learn/pyarrow/python-dotenv/pytest`；CI 只装 5 个包（`market_radar.yml:53-56`）。任何“现有测试/ML 能直接跑”的假设都需要先补环境。

---

## 8. 新项目 `intraday_scalper/` 的边界建议

### 8.1 可以“只读参考”（不建议 import）

| 可复用资产 | 形式 | 理由 |
| :--- | :--- | :--- |
| ATR 自适应崩跌阈值、3 日冷却解锁、断头铡刀公式 | 复制常量/公式 + 单测 | 逻辑自洽（`:1350-1421`），但与“日线级复盘”耦合；可作参考 |
| 风险平价仓位/结构止损公式 | 复制公式 | `v31_strategy.py:479-513`、`integrated_market_radar.py:2752-2758` |
| 令牌桶限速器设计 | 重写（不要 import） | `integrated_market_radar.py:502-560` 与模块级全局状态耦合；新项目应有自己的独立限速器 |
| 市场时段判断逻辑 | 参考 + 补节假日/半日市 | `:612-645` 只有 weekday/hour，不满足日内策略 |
| 缓存 schema 设计 | 参考字段命名，不复用文件 | `.options_cache/.earnings_cache` 由旧模块写，新项目不应共写 |
| 测试断言思路（因果性、purge、三重障碍） | 参考 `test_ml_predictor.py` / `TEST_READY.md` 场景 | 仅参考，不运行旧套件 |

### 8.2 禁止 import / 依赖

| 禁止 | 具体原因 | 证据 |
| :--- | :--- | :--- |
| `import integrated_market_radar` | 模块级会 `load_env_file()`、初始化全局配置、import `create_futu_terminal`/`plotly`；其函数会写 HTML/JSON、起 8765 服务、执行 git push；与旧项目生命周期强耦合 | `:36`、`:154-179`、`:198-205`、`:4488-4514` |
| `import create_futu_terminal` | 会读/写 `data_cache/latest_ai_predictions.json` 与 `predict_radar_5y.html`，还会 import `ml_predictor_5y` 重绘（`:236-253`）；新项目不应改写旧看板 | `:61-69`、`:221-222`、`:236-253` |
| `import ml_predictor_5y` 直接调用 | 4095 行单体、模块级 `load_env_file`、`main()` 会联网下载/训练；无模型持久化；`--realtime` 会发 Finnhub 请求；调用会污染旧缓存 | `:26-51`、`:3998-4091`、`:1202-1266` |
| `import v31_strategy` 作为回测依赖 | 依赖缺失的 `.backtest_5y_cache.pkl`；`load_indicators_cache()` 找不到即抛异常；属于旧两套回测口径之一 | `:287-315` |
| `import backtest_5y_engine` | 会写 `.backtest_5y_cache.pkl` 与 `backtest_report_5y.html`（覆盖既有产物），违反“只读既有文件” | `:526-527`、`:2121-2123` |
| 读写 `data_cache/*.parquet` 作为新项目缓存 | 这些是既有 git 跟踪文件，会被旧流程/用户手动刷新；新项目应有自己的目录与 schema | `git ls-files`；`ml_predictor_5y.py:314` |
| 复用旧 `.options_cache/.earnings_cache/.radar_alert_history/.terminal_notifications` | 由旧模块写入、被 Actions commit；并发读写会互相破坏 | `integrated_market_radar.py:198-204`、`:4500` |
| 触碰旧 HTML/JSON 产物（`index.html`、`futu_pro_terminal.html`、`predict_radar_5y.html`、`macro_radar.html`、`stocks_radar.html`、`data_cache/latest_ai_predictions.json`、点开头 JSON） | Actions 每轮 `git add` 这些文件并 push；本地改动会被 `-X theirs` 覆盖或造成冲突 | `integrated_market_radar.py:4499-4508` |

### 8.3 数据可得性边界（免费 Finnhub 视角，待实测确认）

| 数据需求 | 现有证据 | 新项目应如何处理 |
| :--- | :--- | :--- |
| 实时最新价 | `/quote` 可用（代码在用）：last/change/dp/h/l/o/pc/t | 可作为“价格快照”来源；需自行限速（≤60/min 免费档）与 429 处理 |
| 1 分钟/5 分钟历史 K 线 | 现有代码**无** candle 调用；唯一日内数据来自 yfinance 5m×2d | 先做 candle 权限实测；若 403 → 需付费权限/换供应商/只采集实时（用户确认） |
| 逐笔/实时成交（WebSocket） | 现有代码**无** WS；仓库无 ws 客户端依赖 | 先实测 WS 是否有美股成交推送；不能假设有 |
| 美股节假日/半日市日历 | 现有代码无 | 新项目必须自建（可用 `exchange_calendars` 或静态表），不能复用旧判断 |
| 期权数据 | 现有用 yfinance option_chain（非 Finnhub）；30min 缓存 | 新项目若需期权，需评估 yfinance 稳定性/限频，或改为付费源 |
| 账户/持仓/订单 | 只有手工 `user_portfolio.json`；无券商 API | 若需要自动交易，必须另行接入券商（用户明确授权 + 密钥托管），不属“复用现有” |

> 判定口径：在数据权限实测完成前，`intraday_scalper` **不得把“Finnhub 可提供 1 分钟历史/实时成交”写入设计前提**；应设计成“数据源可插拔 + DATA_UNAVAILABLE 明确降级”，由实测报告决定后续路径。

### 8.4 工作区 / 进程 / 仓库隔离规则（建议写入新项目 README/CONTRIBUTING）

1. **代码隔离**：新项目自包含于 `intraday_scalper/`，有自己的 `requirements.txt` / `pyproject.toml` / `.gitignore`（至少忽略 `.venv/`、`*.pyc`、本地数据目录、`.env`）；不修改根目录 `.gitignore`、`README.md`、workflow、旧脚本。
2. **进程隔离**：新项目不得 import/启动旧模块，不得占用 8765 端口，不得调用 `integrated_market_radar.main()`；不要与旧 daemon 同时运行（旧 daemon 会写 HTML/JSON、修改 git remote、push）。
3. **仓库隔离**：若必须与旧仓库同目录开发，**不要提交到 `main`**（会被机器人 rebase/覆盖）；建议独立分支或独立 clone/仓库；任何 `git commit/push` 必须由人明确授权，且避开自动同步文件。
4. **数据隔离**：新项目输出写到 `intraday_scalper/data/`（自建、自 ignore）或独立数据目录；绝不改写 `data_cache/*.parquet`、`*.html`、`data_cache/latest_ai_predictions.json` 与 4 个点开头 JSON。
5. **密钥隔离**：只用环境变量或 `intraday_scalper/.env`（已被根 `.gitignore:10` 的 `.env` 规则覆盖）；不打印密钥、不把密钥写入被跟踪文件；共享仓库公开，任何持仓/账户信息不得进入提交。
6. **可观测性/限速**：新项目使用自己的 Finnhub 客户端与令牌桶；日志不得打印 token；网络失败要有明确降级/`DATA_UNAVAILABLE` 状态，不得静默伪造数据。
7. **验证方式**：新项目测试必须离线（mock Finnhub/yfinance），不依赖旧 `.backtest_5y_cache.pkl`、不依赖 GitHub Actions 运行状态；跑通 `pytest` 后由 Lead 复核 diff。

### 8.5 建议的下一步（供 Lead 决策）

1. 完成 Finnhub 账户权限实测（引用 `task-2` 的 `data_access_report.md`）：`/quote`、`/stock/candle` 各分辨率、WebSocket、`/stock/profile2` 等；若 candle/WS 不可用，明确选“实时采集 + 本地聚合”还是“换数据供应商（需用户确认）”。
2. 若确实要“1 分钟 VWAP + 实时成交”，设计数据层时先定义 `MarketDataSource` 接口（quote/candle/trade），默认实现只能返回 `DATA_UNAVAILABLE`，禁止用旧 yfinance 5m 冒充 1 分钟数据。
3. 新项目不改动 `.github/workflows/market_radar.yml`；如果未来需要 CI，应新建独立 workflow 文件（属于修改仓库，需要用户明确授权，不在本次只读审计范围）。
4. 若需要“历史回测”，优先用新项目自己的缓存格式重建，而不是依赖缺失的 `.backtest_5y_cache.pkl`；并统一滑点口径（现有 10bps / 1.5bps / README/报告数字互相矛盾，不能直接引用）。

---

## 附录 A：本审计使用的只读命令/脚本（可复核）

- `git log --oneline -8` / `git log --format="%h|%ae|%ai|%s"` / `git shortlog -sne HEAD` / `git rev-list --count HEAD` / `git ls-files` / `git show --stat HEAD` / `git status --porcelain` / `git remote -v`（全部只读）。
- `grep`/`read` 定点取证：`finnhub`、`/quote`、`429`、`backoff`、`sleep`、`SOXL`、`market_session`、`circuit`、`crash`、`halt`、`LightGBM`、`triple_barrier`、`parquet`、`git push`、`firstrade`、`websocket`、`stock/candle` 等。
- Python AST 只读解析：`MONITORED_TARGETS` 标的数量、`FEATURE_COLS` 数量、`EXTENDED_RADAR_SECTOR_MAP`、21 个测试文件的 test 方法计数。
- pandas/json 只读检查：`pd.read_parquet`（两个 parquet 的 shape/columns/dtypes/时间范围/NULL）、`json.load`（8 个 JSON 缓存的顶层结构与字段）。
- 未执行：任何网络请求、任何写文件（除本报告）、任何测试/主程序、任何 git 写操作。

## 附录 B：审计未覆盖/无法验证事项

1. Finnhub 账户实际 tier 与各端点权限（本审计未发请求；以 `task-2` 实测为准）。
2. 两套 5 年回测的真实绩效（缺 `.backtest_5y_cache.pkl`，且未联网重拉）。
3. LightGBM 训练/推理的实际数值结果（环境未安装 LightGBM/sklearn）。
4. 旧 daemon 在 Actions 上的真实运行日志与失败率（未访问 GitHub Actions API）。
5. `.env.example` 之外的本地/服务器密钥配置（未读取任何 `.env` 内容）。
6. 4,577/4,095 行主文件中未读到区段的隐含行为（采用定点 + grep 方法，非逐行全量阅读）。
