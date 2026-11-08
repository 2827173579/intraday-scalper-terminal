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
- 默认引擎：V0.5 A+F（入场仍是 Step3c-A NoCap：10:00–14:00 ET / SMC 严格 AND / 无8% cap；出场 Runner-A：1.5R→+0.20R、2.2R→+1.20R、3.0R→+2.00R，目标 4.5R 或 15:59 EOD）
- 默认风控：B1（$10,000 本金、单笔风险 $60、名义上限 $10,000、碎股 0.01、连亏 3 笔停 15 分钟、单日 -$320 熔断、每日最多 15 笔）
- 3 个月（2026-08-01~11-06，raw IEX）V0.5 A+F + B1：64 笔 / PF 1.2884 / 净 +$573.91 (+5.74%) / MaxDD $635.82；真实 TARGET 4.69%，阶梯锁利 9 笔（均 +$65.68），EOD 22 笔（正16/负6），硬止损 30 笔（46.88%）；全周期 +$884.15，29折 WF +$230.97/14
- Web：本地 FastAPI 之外的 stdlib ThreadingHTTPServer，127.0.0.1:8080；云端 Render 用 0.0.0.0 + $PORT + Basic Auth
- 数据：Alpaca IEX 单交易所（不是 consolidated tape）；盘前/盘后 yfinance 5m prepost + Alpaca snapshot 合成；夜盘 20:00–04:00 不覆盖
- 测试：657 passed

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


---

## 六、换一台没有本地文件的电脑时，怎么说

> 关键前提：GitHub 仓库是**代码唯一来源**；`data/*.parquet`、`.env` 不在仓库里。
> 新电脑只能做：读代码、跑离线测试、改前端、改研究代码、提交 PR。
> **不能**直接复现 3 个月回测，除非你把 `data/` 目录拷过去或重新下载。

### 直接发给 AI 的「远程电脑版」

```text
你在一台全新的电脑上接手项目，本地没有任何项目文件。请以 GitHub 仓库为唯一代码来源，先 clone，不要改代码，先给计划。

仓库：
git clone https://github.com/2827173579/intraday-scalper-terminal
cd intraday-scalper-terminal

项目包在 intraday_scalper/ 子目录，不是仓库根目录：
- 代码：intraday_scalper/
- 测试：intraday_scalper/tests/
- Web：intraday_scalper/web/
- 研究/回测：intraday_scalper/research/、intraday_scalper/tools/
- 部署：仓库根目录 render.yaml / start.sh / Dockerfile

先读：
1. intraday_scalper/CLAUDE.md（工程守则，优先级最高）
2. intraday_scalper/docs/AI_HANDOFF.md
3. intraday_scalper/web/README.md
4. intraday_scalper/research/v04.py、web/terminal_server.py、web/terminal.html

环境（Windows 示例）：
cd intraday-scalper-terminal
python -m venv .venv
.venv\Scripts\pip install -r intraday_scalper/requirements.txt
set PYTHONPATH=.
copy intraday_scalper\.env.example intraday_scalper\.env
（然后由我本人填 Alpaca key，不要把 key 发给你、不要写入代码/日志）

Mac/Linux：
python3 -m venv .venv
.venv/bin/pip install -r intraday_scalper/requirements.txt
export PYTHONPATH=.
cp intraday_scalper/.env.example intraday_scalper/.env

硬性约束：
1. 只修改 intraday_scalper/ 目录内的文件；
2. 不接券商自动下单，只做提醒/风控；
3. 不改默认引擎 V0.4 Step3c-A NoCap 和 B1 风控，除非我明确要求；
4. 绝不提交 .env、API key、密码；不要自动 git commit / push，等我确认；
5. 数据口径保留：Alpaca IEX 是单一交易所，不是 consolidated tape；
6. 改完必须跑：
   Windows: intraday_scalper\.venv\Scripts\python.exe -m pytest intraday_scalper\tests -q
   Mac/Linux: PYTHONPATH=. .venv/bin/python -m pytest intraday_scalper/tests -q
   前端改动还要 node --check 校验内联 script。

重要数据限制（不要自欺）：
- 仓库没有 data/*.parquet（体积大），所以 2026-08~11 的 $10,000 回测、v04c A/B、B1 统计都**不能直接复现**；
- 没有数据时不要运行/伪造回测结果，不要编造 PF/净值；
- 只能跑离线测试（622 个，全部 mock/合成数据，不需要网络和真实数据）；
- 如果要跑真实回测，让我把 data/ 目录用网盘/移动硬盘拷过去，或让我提供数据路径。

工作方式：
- 先给方案，再改代码；一次只做一个任务；
- 每步给：改了哪些文件、为什么、测试命令/结果、剩余风险；
- 不要动 main 分支的默认参数；需要提交时先给我 diff，我说可以再 commit/push。
```

### 新电脑需要提前装什么

- **Git**：用来 clone/push；
- **Python 3.12**：项目要求 Python 3.12；
- **Node.js**（可选）：只用于 `node --check` 校验前端 JS；
- **Alpaca key**：本机 `.env` 自己填；最安全是直接用 Render 上已有的环境变量，不需要把 key 给新电脑上的 AI；
- **如果要回测**：把本机 `intraday_scalper/data/` 整个目录复制过去（约 45MB parquet），或者用 Alpaca 重新拉数据。

### 绝对不要把下面这些发给 AI

- `intraday_scalper/.env`（含 Alpaca key）
- Render 的 `TERMINAL_PASSWORD`
- 你贴过的 Basic Auth 账号密码
- `data/` 里的私有导出数据（如果不想公开）

### 推荐工作流（另一台电脑）

```bash
git clone https://github.com/2827173579/intraday-scalper-terminal
cd intraday-scalper-terminal
git checkout -b dev-<task>
# 改代码 + 跑测试
git add intraday_scalper/...
git commit -m "feat: ..."
git push -u origin dev-<task>
```

然后你自己在 GitHub 上 review/merge 到 `main`；Render 部署的是 `main`，合并后再 Manual Deploy 或等自动部署。不要让 AI 直接 push `main`。
