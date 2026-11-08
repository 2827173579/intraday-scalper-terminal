# `intraday_scalper/web/` — 本地日内终端（可视化 UI）

暗黑 Futu / TradingView 风格的三栏式日内终端，专为 SOXL / SOXS 设计。
**只做展示与提醒，绝不自动下单。**

## 启动

```bash
# 从仓库根目录运行（会连接 Alpaca IEX 实时流）
intraday_scalper/.venv/Scripts/python.exe -m intraday_scalper.web.terminal_server --open

# 浏览器手动打开
http://127.0.0.1:8080/

# 无券商连接 / 收盘后只想看页面（离线演示数据）
intraday_scalper/.venv/Scripts/python.exe -m intraday_scalper.web.terminal_server --demo --open

# 指定端口
... -m intraday_scalper.web.terminal_server --port 8090
```

按 `Ctrl+C` 停止（会同时断开 Alpaca 流）。服务只绑定 `127.0.0.1`，不对外网开放。

## 页面内容

- **顶部状态栏**：ET / KST 双时钟（浏览器每秒刷新）、IEX 连接状态、当前模式
  （DUAL_WATCH / LONG / SHORT）、15:55 ET 强制清仓倒计时进度条。
- **左侧专注目栏**：SOXL / SOXS 两张卡片（实时价、今日涨跌幅、IEX 成交量、
  VWAP、日内高/低、1m/5m bar 数、趋势标签）；点击卡片切换中间图表标的。
- **中间图表区**：本地 canvas 实时蜡烛图（Alpaca IEX 实时 1m/5m/15m/日K，
  与左侧卡片同口径、无延迟，时区 America/New_York）。TradingView 免费版美股有
  15 分钟延迟，已从主图移除（如需参考可在新窗口打开 tradingview.com）。
- **右侧司令塔**：
  1. 风控状态：今日交易笔数 X/15、当日盈亏、单日 -$60 熔断进度、
     连亏冷却倒计时、连亏笔数 / 熔断状态；
  2. 策略雷达：ORB 开盘区间（09:30–09:45 ET）突破状态、SMC 1m/5m
     流动性刺穿观察；
  3. 以损定量执行指南：ENTRY 信号触发时显示建议买入价、硬止损、按 $20
     单笔风险计算的建议股数、2R 止盈价、+1.2R 保本盾状态；
  4. 最近事件流（ENTRY / INFO / RISK_BLOCK / DATA_UNAVAILABLE 等）；
  5. **今日买卖流水 · 人工日志**：录入后的买入/卖出记录、当前持仓、今日已平仓盈亏。

## 买入 / 卖出数据在哪里看

- **信号（还未成交）**：右侧 ③ 卡片出现 `ENTRY · SOXL` 时，会高亮显示建议买入价、
  硬止损价、按 $20 风险算的股数、2R 目标价和信号 ID；同时事件流出现 `ENTRY_SIGNAL`。
- **真实成交（人工确认）**：本系统不会自动下单，成交后需要在 ③ 卡片最下方的输入框录入：

  ```text
  ENTRY <signal_id> <成交价> <股数>   # 记录买入（signal_id 来自 ENTRY 卡片）
  EXIT <signal_id> <成交价>          # 记录卖出
  STOP <成交价>                      # 记录止损卖出
  FLAT                               # 确认空仓
  ```

  录入后 `今日买卖流水 / 当前持仓 / 今日笔数 / 当日盈亏` 会立即更新（写入 SQLite journal）。
- **没有信号时为什么是空的**：三道策略（ORB / VWAP / EMA）条件严格，且 IEX 成交稀疏，
  盘中可能长时间没有 ENTRY；没有信号时执行卡片显示「暂无 ENTRY 信号」属于正常。

## 数据来源与口径

- 实时数据：Alpaca **IEX 单一交易所** WebSocket（与 `main live` 完全同一套
  provider / 数据质量门 / 1m·5m 聚合 / 策略 / 风控 / 信号引擎）；
- 启动时用一次 REST 拉取**前收盘价**与**当日 1m 历史 bar**（用于补全 ORB /
  成交量 / 涨跌幅），之后只走 WebSocket，不做轮询；
- TradingView 图表为 consolidated 参考行情（可能延迟），与左侧 IEX 实时
  卡片口径不同，已在页面底部标注；
- IEX 成交量只占全市场一小部分，PWA/信号数量天然偏少；数据不新鲜时引擎会
  显示 `DATA_UNAVAILABLE` 并停止产生信号（符合 `CLAUDE.md` 约定）。

## 关于 SMC 刺穿

SMC Sweep 只作为**雷达观察指标**：基于已完成 1m/5m bar 的简单因果定义
（最新 bar 刺穿前 20 根摆动低/高点后收回），用于辅助观察；
**它不是生产策略，不产生交易信号，也不参与风控/下单判断**
（生产策略仍只有 ORB / VWAP Reclaim / EMA Pullback）。

## 测试

```bash
intraday_scalper/.venv/Scripts/python.exe -m pytest intraday_scalper/tests/test_terminal_server.py -q
```

覆盖：15:55 倒计时、开盘区间与 ORB 状态、SMC 刺穿识别、5m 聚合、
执行卡（$20 风险 / 2R / 保本盾）、UI state 结构与风控数学。


## 报警大字横幅 + 声音提醒（2026-11-07 追加）

- 右侧 ③ 卡片上方（页面顶部居中）新增**报警横幅**：ENTRY_SIGNAL 为绿色大字，
  STOP_HIT / RISK_BLOCK 为红色脉冲大字，EOD 提醒为黄色，9 秒后自动淡出，可手动关闭；
- 浏览器**声音提醒**（WebAudio 合成，无外部音频文件）：
  - ENTRY / TAKE_PROFIT：三段上升「叮叮叮」
  - STOP / RISK_BLOCK：三段下降「嘟嘟嘟」
  - EOD：两声提示音
- 顶部「🔔 声音开 / 声音关」按钮可随时开关（浏览器本地记忆）；「测试报警」按钮可当场试听 + 试看横幅；
- **浏览器自动播放限制**：首次打开页面后需要随便点一下页面（或点「声音开」/「测试报警」）
  解锁音频；之后新信号会自动响；
- 验证/演示：打开 `http://127.0.0.1:8080/?alert_demo=1` 会显示一条演示横幅（不发声）。

声音只在新事件出现时触发一次；首次加载页面只显示最近一条警报横幅、不补响历史事件。


## 一键启动 / 停止（Windows 双击即可）

不用打开命令行，直接双击 `intraday_scalper/web/` 下这三个文件之一：

| 文件 | 作用 |
|---|---|
| **start_terminal.bat** | 启动服务并自动打开浏览器；如果服务已在运行则只打开浏览器 |
| **stop_terminal.bat** | 停止后台服务（会清理终端相关 python 进程） |
| **terminal_control.bat** | 菜单版：1 启动 / 2 停止 / 3 只开浏览器 / 0 退出 |

说明：

- 启动脚本会在**后台最小化窗口**里运行服务（窗口标题 `SOXL-SOXS-TERMINAL`），
  关掉这个窗口或双击 `stop_terminal.bat` 都会停服；
- 服务只监听 `127.0.0.1:8080`，不对外网开放；
- 周末/节假日打开时，页面显示**最近一个交易日**的数据（例如周六显示周五收盘），
  并且顶部阶段显示 `CLOSED`；下一个交易日开盘后会自动切换到实时盘中数据；
- 如果浏览器打不开，先双击 `stop_terminal.bat` 再双击 `start_terminal.bat`。


## V0.4 引擎接入（2026-11-08）

- **盘后/周末自动回显最近交易日**：优先读取本地 `data/{SOXL,SOXS}_1m_raw.parquet`
  最近一个完整 session（当前为 2026-11-06 周五：SOXL 391 根 / SOXS 385 根），
  REST 只在本地数据不是最新时补齐；因此收盘后左侧 Bars 不再是 0/0，图表完整显示周五走势。
- **复盘标记**：主图直接绘制 V0.4 的 `ENTRY`（绿▲）/ `STOP`（红▼）/ `TARGET`（蓝○）/
  `BREAKEVEN`（黄◇）/ `EXIT`（青□）点位；数据来自 `var/backtest/v04_friday_markers.json`。
- **复盘卡**：右侧新增「④ 最近交易日复盘」+ ③ 卡内嵌摘要：
  周五 V0.1/V0.3/V0.4 三版本笔数/净盈亏、全周期 5 步消融（PF/净/ WF 净）与回测警告。
- **盘中 V0.4**：交易时段内每完成 5m bar，服务端用
  `research.v04.evaluate_v04_session` 在本地历史 + 当日实时 bars 上重放，
  产生的 marker 会驱动报警横幅/声音（不再转发 V0.1 生产策略的 ENTRY/STOP）。
- **重要警告**：V0.4 final（含 +1.2R 分批止盈）在 raw 29 折 WF 上 **PF 0.951 / WF -$516.83**，
  比 V0.3 更差；Step1（仅 10:00–14:00 入场窗）是样本外相对最好的变体（WF -$176.59 / 17 折），
  Step3（无分批止盈）全周期 +$9.28 但 WF 仍 -$250.63。**当前 Web 按需求展示 V0.4 final，
  仅作观察，不建议实盘**；若要把生产默认切到 Step1/Step3，需要你确认。


## V0.4b 默认引擎 = Step3b（2026-11-08 更新）

- 已废弃 Step4（+1.2R 分批止盈）：全周期 -$364.56 / WF -$516.83，属于负贡献。
- 默认改为 **Step3b**：10:00–14:00 入场窗 + VWAP趋势过滤 + SMC_SWEEP + 两项固定过滤
  （EMA_PULLBACK 日内涨幅≥8% 禁止追高；SMC 要求 close≥EMA21 或 close≥session VWAP）。
- 回测对比（raw 29 折 WF）：
  | 变体 | 全周期 净/PF | Holdout 净/PF | WF 净/盈利折 |
  |---|---|---|---|
  | Step1 | -$10.00 / 0.999 | +$274.03 / 1.061 | **-$176.59 / 17** |
  | Step3 | +$9.28 / 1.001 | +$385.53 / 1.082 | -$250.63 / 15 |
  | **Step3b（默认）** | **+$76.32 / 1.012** | **+$566.85 / 1.139** | -$210.04 / 15 |
  说明：Step1 的 WF 净与盈利折更优（-176.59 / 17），但全周期 -$10、holdout 仅 +$274；
  Step3b 全周期与 holdout 均为三者最好。若要把默认切 Step1/Step3，改 `terminal_server.py`
  里 `_maybe_evaluate_v04` 的 `variant=` 与 replay 选变体一行即可。
- 周五 2026-11-06：Step3b 仅 1 笔（11:25 SOXL SMC 接刀 STOP -$20.51）；
  12:40 SOXS EMA 山顶追高已被 8% 防追高过滤拦截（Step3 的 -$21.63 被避开）。
  SMC 接刀未避开：按任务规定的 OR 规则，close=46.23 ≥ EMA21=45.60 即放行（仅低于 VWAP 46.65）；
  如需"同时站上 EMA21 和 VWAP"的新规则（Step3c）需重新审批/预注册后跑 WF。
- Web 复盘卡新增 Step1/Step3/Step3b 对比表（全周期/Holdout/WF 净与盈利折），
  `v04b_compare.md`、`v04b_report.json` 为完整证据。


## 盘前/盘后扩展时段（Extended-Hours，2026-11-08）

- 数据：`providers/extended_hours_provider.py` 通过 yfinance `interval="5m", prepost=True, period="5d"`
  拉取 04:00–20:00 ET 的 5m bars；盘前/盘后成交量多为 0，PM_VWAP 用典型价均值（页面已注明）。
- 指标：5m EMA9 / EMA21 / ATR(14) 跨盘前→盘中→盘后连续滚动（warm-up 含历史 session），
  不再在 09:30 冷启动。
- 锚点：`PM_High` / `PM_Low` / `PM_VWAP`（04:00–09:30 ET），左侧卡片与主图都会画出 LM_High/Low 水平线。
- 信号（仅提醒、限价单）：
  - `SMC_PM_LOW_SWEEP`：盘中刺穿 PM_Low 后收回，且 close ≥ EMA21 & session VWAP（高优先级）；
  - `PM_HIGH_BREAKOUT`：突破 PM_High 后回踩 EMA9/VWAP 不破；
  - `EMA_PULLBACK` / `SMC_SWEEP`：07:30–09:30 ET（盘前）与 16:00–17:30 ET（盘后）窗口，
    卡片/事件标注【盘前信号·限价单】/【盘后信号·限价单】（Firstrade 勾选 Day + Ext）。
- Web：`/api/extended` 返回 bars/phases/PM 锚点/signals；主图开启“含盘前盘后”时用浅色
  背景区分 pre/post，09:30 与 16:00 画竖虚线，并绘制 PM_High / PM_Low 水平参考线与扩展时段信号标记。
- 限制：yfinance prepost 不是实时 IEX 逐笔，盘前/盘后已收盘 K 线可能有延迟；夜盘 20:00–04:00
  没有公共 5m 数据，不做自动触发（04:00 后立即捕捉跳空并纳入盘前 K 线）。


## V0.4c Step3c-A NoCap = 当前默认（2026-11-08 更新）

- 默认引擎：**Step3c-A NoCap**——SMC 右侧确认从 OR 升级为严格 **AND**（close ≥ EMA21 **且** close ≥ session VWAP），
  并按 A/B 结果 **移除 8% 防追高锁**。
- A/B 数据（raw 483,751 根 / 29 折 WF）：
  | 变体 | 全周期 笔/PF/净/MaxDD | Holdout PF/净 | WF 净/PF/盈利折 |
  |---|---|---|---|
  | Step3b（OR+8%） | 730/1.012/+$76.32/$702 | 1.139/+$566.85 | -$210.04/0.961/15 |
  | **NoCap（AND，默认）** | **751/1.009/+$64.57/$520** | 1.086/+$388.18 | **-$131.65/0.978/16** |
  | WithCap（AND+8%） | 681/1.002/+$13.55/$597 | 1.113/+$446.34 | -$181.78/0.965/17 |
- 选择理由：NoCap 的 **WF 净/PF 三者最优、MaxDD 最低、2024 段最好**，全周期仍为正；8% 锁只在周五避免 1 笔
  (-$21.63) 但代价是 2024/全周期/WF 都变差（WithCap），故移除。
- 周五 2026-11-06：AND 已挡掉 11:25 SOXL SMC 接刀；12:40 SOXS EMA 山顶追高（+15.12%）在 NoCap 下恢复并亏损 -$21.63
  （Step3b 为 -$20.51，WithCap 为 0 笔）。
- Web：默认变体、复盘卡、图表标记均已改为 NoCap；对比表显示 Step3b / NoCap / WithCap 三行。


## 低延迟混合行情（2026-11-08 追加）

- 盘前/盘后基础 5m K 线：yfinance `prepost=True`（有 10–15 分钟延迟）；
- **08:00–17:00 ET 混合补丁**：Web 服务每 **20 秒** 调 Alpaca IEX Snapshot
  （latest trade / quote mid），把最新价合并进**当前未闭合 5m bucket**（high/low/close），
  并重算 PM_High / PM_Low；yfinance 基础数据每 5 分钟才刷新一次；
- 顶部新增 **数据延迟** 标记：`数据延迟 Xs · Alpaca IEX 实时`（<30s 绿色 / <300s 黄色 / 否则红色）；
  无实时 quote 时回退显示 `yfinance 5m` 的 K 线年龄；
- `/api/extended` 返回 `quotes` / `data_age_seconds` / `bar_ages_seconds` / `quote_source`；
- 限制：Alpaca 免费 IEX 在盘前/盘后成交稀疏，合并只能更新"最新价"；20:00–04:00 夜盘无数据，
  不做自动触发；Snapshot 轮询仅用于显示/研究，不改变 V0.4 的 1m/5m 因果信号规则。


## B1 风控默认（2026-11-08 切换）

- Web 实时仓位计算器 / 复盘看板默认切到 **B1**：
  `$10,000` 本金、单笔固定风险 `$60`、名义上限 `$10,000`、碎股 0.01、
  连亏 3 笔停 15 分钟、单日 `-$320` 熔断（`terminal_server.b1_config()`）。
- 卡片① 显示 `单日 -$320 熔断进度`、`连亏冷却（3 笔后 15 分钟）`、
  `B1 · 本金 / 单笔风险 / 名义上限 $10,000 / $60 / $10,000`；
  执行卡建议股数按 B1 计算。
- 复盘看板不再单用名义胜率：`/api/replay` 新增 `period_b1`（3 个月 B1 汇总）
  与 `exit_structure`（TARGET / SHIELD_STOP / EOD_POSITIVE / EOD_NEGATIVE / INITIAL_STOP），
  卡片③内嵌一行「真实出场」，卡片④展示完整表格。
- 3 个月 B1 验证：68 笔、PF **1.2138**、净 **+$385.89 (+3.86%)**、MaxDD **$703.82**；
  名义胜率 51.47%，但**真实 TARGET 2R 率仅 23.53%**；保本盾 10 笔、
  EOD 14 笔（正 10/负 4）、硬止损 28 笔（41.18%）。
