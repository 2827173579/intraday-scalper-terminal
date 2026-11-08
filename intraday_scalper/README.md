# SOXL / SOXS Intraday Scalper

> 独立于根目录 `integrated_market_radar.py` / `ml_predictor_5y.py` 的分钟级日内信号系统。
> **只做分析、信号、风控计算、声音提醒与人工交易记录，绝不自动下单。**

---

## 🗄️ 项目状态（2026-11-06）：策略研究已归档，仅作研究/回测基础设施

> 在 raw（未复权）IEX 1m 真实数据 + 严格 29 折 Walk-Forward 下，V0.1/V0.2/V0.3
> 全周期净盈亏分别为 **-$1,184.41 / -$736.64 / -$214.76**，V0.3 样本外 **-$234.92**；
> 结构敏感性 9 格网格全部为负（详见 [`docs/STRATEGY_ARCHIVE.md`](docs/STRATEGY_ARCHIVE.md)
> 与 [`docs/verification_report.md`](docs/verification_report.md) §9–§13）。
>
> **结论：没有任何版本具备正期望，不可实盘。** 本仓库保留数据管道、回测/归因工具与测试套件
> 作为基础设施；重新开始策略研究前必须换用 consolidated 数据并预注册 kill criteria。

---

## ✅ 数据源现状（2026-11-06 更新）

**主数据源：Alpaca Markets Paper + IEX（已接入并通过实测）**

| 能力 | 状态 |
|---|---|
| REST 最新 1m bar / 报价 / 成交 | ✅ HTTP 200（SOXL / SOXS） |
| WebSocket `wss://stream.data.alpaca.markets/v2/iex`（trades + 1m bars） | ✅ 认证与订阅成功（盘前无成交属正常，09:30 ET 后开始推送） |
| 历史 1m bars（回测/研究） | ✅ IEX 口径，可回溯到 2022 年（分页拉取） |
| 免费档限制 | WebSocket 最多 30 个订阅符号；限速 200 次/分钟 |

> ⚠️ **IEX 口径**：IEX 是单一交易所，成交量只占美股全市场的一小部分（约 2%–4%），
> 由它聚合的 VWAP 是「IEX-trade VWAP」，**不是全市场 consolidated tape**。
> 因此：成交量类过滤阈值需要按 IEX 口径理解，报告/信号里不得宣称"全市场真实成交量"。

**备用数据源：Finnhub 免费档（不可用于日内引擎）**

| 能力 | 状态 |
|---|---|
| REST `/quote` | ✅ 可用，但只是上一交易日收盘快照 |
| REST `/stock/candle`（1m/5m/D…） | ❌ HTTP 403（无历史 K 线权限） |
| WebSocket `wss://ws.finnhub.io` | ✅ 可连接；❌ 订阅 SOXL/SOXS 收不到实时美股成交 |

程序在无有效数据时显示 `DATA_UNAVAILABLE` 并停止产生交易信号——这是设计行为，
不是 bug。详见 [`docs/alpaca_probe_result.md`](docs/alpaca_probe_result.md) 与
[`docs/data_access_report.md`](docs/data_access_report.md)。

---

## 快速开始

```bash
# 1) 依赖（venv 已建好；重建时：）
cd intraday_scalper
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .   # 依赖见 pyproject.toml（含 alpaca-py）

# 2) 密钥（gitignored，勿提交）：复制 .env.example 为 .env 并填写
#    ALPACA_API_KEY / ALPACA_SECRET_KEY / ALPACA_DATA_FEED=iex
#    FINNHUB_API_KEY（可选，仅备用）
cp .env.example .env

# 3) 测试（离线，不需要网络）
.venv/Scripts/python.exe -m pytest tests -q

# 4) Alpaca 连通性检查（REST bar/quote/trade）
.venv/Scripts/python.exe -m intraday_scalper.main check

# 5) 实时模式（auto：有 Alpaca 凭证就用 Alpaca IEX）
.venv/Scripts/python.exe -m intraday_scalper.main live --mode dual_watch
#       强制指定数据源：--provider alpaca | finnhub

# 6) 合成数据演示（验证看板/引擎链路，不是真实行情）
.venv/Scripts/python.exe -m intraday_scalper.main demo

# 7) 合成数据回测（逻辑验证）
.venv/Scripts/python.exe -m intraday_scalper.main backtest

# 8) 真实 IEX 历史 1m bars 下载到本地 parquet（2024-01-01 至今，常规时段）
.venv/Scripts/python.exe -m intraday_scalper.tools.download_history

# 9) 真实数据全周期回测 + walk-forward 战报（终端打印 + var/backtest/*.json）
.venv/Scripts/python.exe -m intraday_scalper.tools.run_real_backtest
```

> Windows 下入口：`intraday_scalper\.venv\Scripts\python.exe -m intraday_scalper.main <cmd>`
> 需要仓库根目录在 `sys.path`（从仓库根运行即可）。

---

## 人工交易记录（不连接券商）

程序只**提醒**，成交由用户人工完成并在终端录入：

```text
ENTRY <signal_id> <price> <shares>    # 录入实际买入价格与数量
EXIT  <signal_id> <price>             # 录入实际卖出价格
STOP  <signal_id> <price>             # 记录止损
CANCEL <signal_id>                    # 取消未执行信号
FLAT                                  # 确认当前空仓
```

信号价与实际成交价分别保存，自动统计**真实执行滑点**。

---

## 风控参数（`config/default.yaml` 可改，未回测验证前不得视为有效）

| 参数 | 默认值 |
|---|---|
| 参考账户资金 | $3,000 |
| 单笔最大风险 | $20 |
| 单笔最大名义仓位 | $1,200 |
| 单日最大净亏损（熔断） | $60 |
| 连续两笔止损冷却 | 20 分钟 |
| 每日最多开仓 | 15 次（允许 0 交易） |
| 15:30 ET 后 | 禁止新开仓 |
| 15:50 / 15:55 ET | 提醒检查持仓 / 高优先级连续提醒（只提醒人工平仓） |

仓位公式：`shares = min(MAX_RISK/|entry-stop|, MAX_NOTIONAL/entry)`，
整数股向下取整（碎股默认关闭），显示计划股数/名义/理论风险/预估滑点/计入成本风险。

---

## 架构

```text
intraday_scalper/
├── core/          # 跨模块契约：models / timeutil / config（UTC、会话、DST）
├── providers/     # Alpaca IEX（首选，实时 trades+1m bars）、Finnhub WS/REST（备用）、数据质量
├── aggregation/   # tick→1m OHLCV、1m→5m、真实 VWAP、开盘区间
├── features/      # EMA / RSI / ATR / OpeningRange 等纯函数（无未来函数）
├── strategies/    # ORB(5m/15m)、VWAP Reclaim、EMA Pullback + registry
├── risk/          # 仓位计算 + 风控状态机（熔断/冷却/互斥/最大次数/EOD）
├── signals/       # 信号引擎、双轨模式状态机、持仓提示
├── journal/       # SQLite 人工交易日志 + ENTRY/EXIT/STOP/CANCEL/FLAT
├── dashboard/     # Rich 终端看板（ET/KST、VWAP/EMA/RSI、风控预算）+ winsound 提醒
├── backtest/      # 事件驱动回测、绩效指标、walk-forward、合成数据生成器
├── research/      # V0.2 研究版：5m 闭合信号 + 镜像共振 + IEX 口径（不污染生产策略）
├── data/          # 原始 tick / bar 落盘 + 真实历史 bar parquet（var/ 与 *.parquet 已 gitignore）
├── tools/         # download_history、run_real_backtest、finnhub_probe、test_alpaca_feed
├── config/        # default.yaml、.env.example
├── docs/          # AUDIT.md、data_access_report.md、alpaca_probe_result.md 等证据
├── tests/         # 离线单元/集成测试
└── var/           # 运行时数据、日志、SQLite、回测输出（不提交）
```

## 阶段状态

- [x] 阶段一：现有项目只读审计（[`AUDIT.md`](AUDIT.md)）
- [x] 阶段一：Finnhub 数据权限实测（[`docs/data_access_report.md`](docs/data_access_report.md)）
- [x] WebSocket 客户端 + 断线重连 + 数据质量监控
- [x] 1m / 5m K 线聚合 + 真实 VWAP
- [x] ORB / VWAP Reclaim / EMA Pullback 三策略
- [x] 风控状态机 + 人工交易日志
- [x] Rich 看板 + 声音提醒
- [x] 事件驱动回测 + 指标 + walk-forward（合成数据）
- [x] **Alpaca IEX 实时数据源接入（provider + 主程序 wiring + live smoke）**
- [x] 真实 IEX 历史 1m bars 可用（2022 年至今，单交易所口径）
- [x] **真实历史数据本地化（2024–2026，714 个交易日 / 467,120 根 1m bar）**
- [x] **真实数据全周期回测 + 风控统计**（引擎时钟 bug 已修复）
- [x] **真实数据 Walk-Forward 样本外验证（29 折，train 120d/test 20d/step 20d）**
- [x] **V0.1 逐笔归因（时段 / 策略×标的 / 出场 / MAE-MFE），见 docs §9**
- [x] **V0.2 结构性降噪（5m 闭合信号 + 镜像共振 + IEX 相对口径），见 docs §10**
- [x] **V0.3 结构重构 + 5 步累计消融（碎股 / 移除 ORB / 14:00 截止 / +1.2R 保本盾），见 docs §11**
- [x] **raw（未复权）价格交叉验证：adjusted 数据存在 16,631 根 SOXS 覆盖缺失，V0.3 结论以 raw 为准，见 docs §12**
- [ ] 全市场 consolidated（SIP）数据（免费档不具备；如需请升级/换源）
- [x] **raw canonical 基线重跑（V0.1/V0.2/V0.3）+ V0.3 结构敏感性网格（9 格全负），见 docs §13**
- [ ] 策略有效性与实盘条件（raw canonical 下 V0.3 全周期 -$214.76 / WF -$234.92，**不可实盘**）

## 验证结果（详见 [`docs/verification_report.md`](docs/verification_report.md)）

```text
pytest intraday_scalper/tests -q            ->  556 passed / 0 failed  (0 skipped)
main check                                  ->  ALPACA OK（bars/quotes/trades 均 HTTP 200）
main live --provider alpaca --max-frames 2   ->  连接 CONNECTED，盘前无成交不产生信号（符合预期）
download_history（2024-01-01~2026-11-06）    ->  SOXL 254,479 + SOXS 212,641 根；714 交易日；0 重复/空值
run_real_backtest（全周期）                  ->  2,687 笔 / 胜率 34.72% / PF 0.91 / 净 -$1,734.06
                                              最大回撤 $2,265（69.19%）/ -$60 熔断 22 天
run_real_backtest（Walk-Forward 29 折）     ->  样本外 2,341 笔 / 胜率 34.05% / PF 0.89
                                              净 -$1,956.23 / 平均 R -0.105 / 9/29 折盈利
run_v02_backtest（V0.2 全周期）             ->  1,206 笔 / 胜率 38.89% / PF 0.967 / 净 -$385.40
                                              最大回撤 $1,417 / 日均 1.95 笔 / 滑点 $384
run_v02_backtest（V0.2 WF 29 折）           ->  样本外 1,035 笔 / 胜率 37.58% / PF 0.918
                                              净 -$852.52 / 平均 R -0.041 / 11/29 折盈利
run_v03_backtest（5 步消融，Step4=V0.3）    ->  945 笔 / 胜率 43.81% / PF 1.010 / 净 +$90.25
                                              最大回撤 $509.30 / 保本盾 arm 357 次
run_v03_backtest（V0.3 WF 29 折）           ->  样本外 805 笔 / 胜率 44.10% / PF 0.994 / 净 -$45.75（adjusted 数据）
                                              16/29 折盈利 / holdout 2025-2026 净 -$56.62
run_v03_raw_crosscheck（raw 未复权）        ->  raw_full Step4：970 笔 / PF 0.977 / 净 -$214.76
                                              WF 样本外 -$234.92 / PF 0.970 / 16/29 折盈利
run_raw_canonical（raw canonical 基线）     ->  V0.1 -$1,184.41 / V0.2 -$736.64 / V0.3 -$214.76
                                              （笔数 3,318 → 1,637 → 970；2024 正收益未在 holdout 复现）
run_v03_sensitivity（9 格结构网格）         ->  全周期 -$110 ~ -$298、WF -$151 ~ -$294，9/9 格全负
git status --porcelain                      ->  仅 "?? intraday_scalper/"（既有文件零修改）
```

> ⚠️ 最终结论（真实 Alpaca IEX、未做参数调优；canonical = **raw/未复权**，见 docs §12/§13）：
> raw canonical 下：V0.1 **-$1,184.41**（PF 0.952）→ V0.2 **-$736.64**（0.957）→ V0.3 **-$214.76**（0.977）；
> V0.3 的 29 折 WF 样本外为 **-$234.92（PF 0.970，16/29 盈利折）**；
> V0.3 结构敏感性 9 格网格（截止 13:30/14:00/14:30 × 保本偏移 0.02/0.05/0.10R）**全部为负**
> （全周期 -$110~-$298，WF -$151~-$294）→ 结构改进方向真实，但**没有任何版本具备正期望，不可实盘**。
> 历史 adjusted 数据上的 V0.3 +$90.25 已被证明是 16,631 根 SOXS bar 缺失导致的覆盖 artifact。
> 详见 [`docs/verification_report.md`](docs/verification_report.md) §9–§13（§12/§13 为最终口径）。

> 💡 Windows 终端若中文显示乱码，可先执行 `chcp 65001` 或设置 `PYTHONUTF8=1`。

## 免责声明

本项目仅供量化研究与学习，不构成投资建议。策略未经真实数据验证前**不得**
描述为稳定盈利系统。先证明行情数据可靠，再证明策略有统计优势，最后才考虑实盘。
