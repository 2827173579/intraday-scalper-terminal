# 交给其他 AI（Antigravity Argon / Gemini 等）继续开发时，直接粘贴这份

> ⚠️ 先看安全：**不要把 `.env`、Alpaca key、Render 密码贴给任何外部 AI**。只给仓库地址和本地路径；密钥由你自己在 Render Environment / 本地 `.env` 里填。

---

## 一、最短可用版（先发这段，让它读项目）

```text
你接手一个 SOXL / SOXS 分钟级日内信号终端项目，先不要改代码，先读文档并给我计划。

项目位置（本地）：
E:\wendang\deepseek-harness\default-workspace\intraday_scalper
GitHub（公开，可 clone）：
https://github.com/2827173579/intraday-scalper-terminal
线上（Render，Basic Auth，密码在 Render 环境变量里，不要问我要明文）：

请先读：
1. intraday_scalper/CLAUDE.md（工程守则，优先级最高）
2. intraday_scalper/web/README.md
3. intraday_scalper/research/v04.py（当前引擎）
4. intraday_scalper/web/terminal_server.py 和 web/terminal.html
5. intraday_scalper/tools/run_v04c_ab.py、tools/analyze_exit_structure_10k.py

当前状态（以代码和测试为准）：
- 默认引擎：V0.4 Step3c-A NoCap（10:00–14:00 ET 入场窗；SMC 必须同时站上 EMA21 和 session VWAP；无 8% 防追高；无 Step4 分批止盈）
- 默认风控：B1（$10,000 本金、单笔风险 $60、名义上限 $10,000、碎股 0.01、连亏 3 笔停 15 分钟、单日 -$320 熔断、每日最多 15 笔）
- 3 个月（2026-08-01~11-06，raw IEX）B1：68 笔 / PF 1.2138 / 净 +$385.89 / MaxDD $703.82；真实 TARGET 2R 率 23.53%，保本盾 10 笔，EOD 14 笔（正 10/负 4），硬止损 28 笔（41.18%）
- Web：本地 FastAPI 之外的 stdlib ThreadingHTTPServer，127.0.0.1:8080；云端 Render 用 0.0.0.0 + $PORT + Basic Auth
- 数据：Alpaca IEX 单交易所（不是 consolidated tape）；盘前/盘后 yfinance 5m prepost + Alpaca snapshot 合成；夜盘 20:00–04:00 不覆盖
- 测试：622 passed

先回答我：
A. 你对当前代码结构的理解；
B. 你准备怎么改（文件清单 + 风险点）；
C. 需要我补充哪些信息。
先不要动代码，等我确认。
```

---

## 二、真正改功能时（把第一段换成任务）

```text
在刚才的项目上实现：<把你的需求写清楚，例如"给真实持仓加止损/止盈提醒">

硬性约束（必须遵守）：
1. 只允许修改 intraday_scalper/ 目录内的文件；不要动仓库根目录其他项目。
2. 严禁接券商自动下单；系统只做提醒、风控、人工下单。
3. 不要改默认引擎和默认风控（V0.4 Step3c-A NoCap + B1），除非我明确要求。
4. 不要把 API key / 密码写进代码、日志、提交；不要提交 .env。
5. 不要自动 git commit / push，除非我明确说可以。
6. 数据口径必须保留："Alpaca IEX 单一交易所，不是全市场 consolidated tape"；盘前/盘后数据源要标注。
7. 改完必须自己跑：
   intraday_scalper\.venv\Scripts\python.exe -m pytest intraday_scalper\tests -q
   并且 JS 改动用 node --check 校验（terminal.html 内联 script）。
8. 先给方案再改代码；每次只改一个可验证的小步；报告里写清楚改了哪些文件、为什么、测试结果。
9. 如果改动会影响回测数字/默认参数，先告诉我，不要偷偷调参。

当前已知限制（不要重复踩）：
- 根目录 data/*.parquet 很大，没有进 GitHub 公开仓库；做 3 个月回测需要本地数据或我另外提供。
- Render 免费实例会休眠；Alpaca 免费档同一时间只能一个 IEX WebSocket，本地和云不要同时跑。
- "加仓/金字塔"没有经过 WF 验证，不要直接加；先做研究并给我 A/B 结果。
- 盘前/盘后信号目前只是提醒；PM_High/PM_Low 叠加回测在 2026-08~11 是负贡献，不要把它当已验证 alpha。
- 现有周五复盘只有 2 笔交易，图上 ENTRY 多、STOP/TARGET 少是正常的，不要误以为出场逻辑坏了。

交付格式：
1. 修改文件列表 + 关键 diff 说明；
2. 测试命令和结果；
3. 回测/线上验证结果（如涉及）；
4. 剩余风险和你建议的下一步。
```

---

## 三、如果你要它改 Web / 部署

```text
部署环境：
- Render Blueprint 仓库根目录 render.yaml；start.sh 绑定 0.0.0.0:$PORT；健康检查 /healthz；Basic Auth 用 TERMINAL_PASSWORD。
- 线上服务：intraday-scalper-terminal.onrender.com（账号密码不要写进 prompt）。
- 静态快照：https://2827173579.github.io/intraday-scalper-terminal/
- 修改前端后，先在本地起服务看效果，再 commit；Render 需要 Manual Deploy → Deploy latest commit 才生效。
- 不要为了"方便"把 TERMINAL_PASSWORD 删掉；公网必须保留鉴权。
```

---

## 四、给 Gemini / Argon 的英文版（可选）

```text
You are taking over a SOXL/SOXS intraday signal terminal. Read these first, do not change code yet:
- Local: E:\wendang\deepseek-harness\default-workspace\intraday_scalper
- GitHub: https://github.com/2827173579/intraday-scalper-terminal
- Read: intraday_scalper/CLAUDE.md, web/README.md, research/v04.py, web/terminal_server.py, web/terminal.html

Current defaults:
- Engine: V0.4 Step3c-A NoCap (10:00-14:00 ET entry window; SMC needs close >= EMA21 AND session VWAP; no 8% cap; no Step4 scale-out)
- Risk: B1 ($10k equity, $60 risk/trade, $10k notional cap, fractional 0.01, 3 consecutive stops -> 15 min cooldown, -$320 daily breaker, max 15 entries/day)
- 3-month raw IEX B1: 68 trades, PF 1.2138, net +$385.89, maxDD $703.82; true TARGET 23.53%, shield 10, EOD 14 (10 pos/4 neg), hard stops 28 (41.18%)
- Tests: 622 passed. Web served by stdlib ThreadingHTTPServer, local 127.0.0.1:8080, Render 0.0.0.0:$PORT + Basic Auth.

Hard rules:
1. Only modify files under intraday_scalper/.
2. Never place broker orders; alerts only, manual execution.
3. Keep default engine/risk B1 unless I explicitly ask.
4. Never commit .env or secrets; do not auto git commit/push without permission.
5. Preserve the IEX single-venue data caveat (not consolidated tape) and data-source labels.
6. Run `intraday_scalper\.venv\Scripts\python.exe -m pytest intraday_scalper\tests -q` after changes; node --check for inline JS.
7. Report: plan first, then files changed, tests, risks, next step. No silent parameter tuning.

Known constraints: data parquet files are not in the public repo (3-month backtest needs local data); Render free sleeps; Alpaca free allows only one IEX WebSocket (do not run local and cloud simultaneously); PM_High/PM_Low overlay was negative in Aug-Nov 2026; no unvalidated pyramiding/add-ons.
```

---

## 五、简短结论

- **先让它读 `CLAUDE.md` + 仓库**，再让它给计划；不要一上来就改。
- 每次只派一个明确任务，并把"不要改默认引擎/风控、不要接自动下单、不要提交密钥"写死。
- 任何回测/参数改动，要求它给 **A/B + 29 折 WF + 测试结果**，否则不要合入。
