# CLAUDE.md — intraday_scalper 开发守则

本文件是 `intraday_scalper/` 子项目的工程守则，优先级高于个人偏好。
任何自动化代理（Claude Code / DSH agent / 其他）修改本目录时必须遵守。

## 0. 边界（最重要）

- **只允许修改 `intraday_scalper/` 目录内的文件。**
- 严禁修改、覆盖、删除仓库根目录的既有文件，尤其是：
  `README.md`、`integrated_market_radar.py`、`ml_predictor_5y.py`、
  `requirements.txt`、`.gitignore`、`.github/workflows/*`、根目录所有
  `test_*.py`、`backtest_5y_engine.py`、`v31_strategy.py` 及数据缓存文件。
- 严禁中断仓库根目录项目的后台进程 / GitHub Actions。
- 本子项目**不自动 commit / 不自动 push**，除非用户明确要求。
- 严禁接入券商自动下单（Firstrade 或任何券商 API）。系统只做分析、
  信号、风控计算、声音提醒与**人工**交易记录。

## 1. 项目定位

SOXL / SOXS 分钟级日内信号系统。1 分钟为主、5 分钟辅助，持仓 1–30 分钟，
人工下单。一天最多 15 次开仓（无最低次数要求，允许全天 0 交易）。

## 2. 数据事实（不要自欺）

- **主数据源：Alpaca Markets Paper + IEX**（`providers/alpaca_provider.py`）。
  REST 最新 bar/quote/trade 与 WebSocket trades+1m bars 均已实测可用（2026-11-06）。
- **IEX 是单一交易所**：成交量只占美股全市场一小部分（约 2%–4%），
  由此得到的 1m 成交量与 VWAP 都是 IEX 口径，**不是全市场 consolidated tape**；
  报告与信号中必须标注口径，不得宣称"全市场真实成交量/VWAP"。
  免费档 WebSocket 订阅上限 30 个符号；IEX 无盘前成交（09:30 ET 后才有实时流）。
- **备用 Finnhub 免费档不可用于日内引擎**：`/quote` 只是上一交易日收盘快照；
  `/stock/candle` 全部 403；WebSocket 订阅美股没有成交数据。仅保留作 fallback。
- 没有可靠实时数据时必须显示 `DATA_UNAVAILABLE` 并停止产生信号，
  禁止拿 `/quote` 的 high/low 冒充 1 分钟 K 线，禁止无成交量时伪造 VWAP。
- 禁止每 2 秒轮询 REST `/quote`；REST 仅在启动时做校验
  （Finnhub 每交易日最多 2 次；Alpaca 只做一次性 probe）。
  禁止多 API Key 绕过额度。
- 回测数字必须标注数据口径：合成数据 = 仅逻辑验证；IEX 数据 = 单交易所口径、
  样本不足时明确写"不完整"，不得把 IEX 回测结果当作全市场策略有效性证据。

## 3. 数据模型约定

- 所有时间戳必须是 **timezone-aware UTC**（`core/models.py` 会强制校验）；
  展示层再转 `America/New_York` / `Asia/Seoul`（`core/timeutil.py`）。
- 1m bar 只能由已完成分钟内的成交聚合；5m bar 只能由**已完成的 1m bar**聚合。
- VWAP 必须由 `Σ(price*volume)/Σvolume` 得到（真实成交加权）。
- 任何"未完成 K 线"不得用于生成确认信号。
- 每个跨模块类型都定义在 `core/models.py`，其他模块禁止重复定义。

## 4. 策略与风控

- 第一阶段只允许三个确定性策略：ORB（5m/15m）、VWAP Reclaim、EMA Pullback。
- SMC Sweep / MSS 暂不实现（缺乏无未来函数的精确定义）。
- 双轨模式：`long_soxl` / `long_soxs` / `dual_watch`；任何时候最多一个持仓，
  切换必须重新满足完整入场条件并受冷却限制。
- 风控参数在 `config/default.yaml`，禁止硬编码到策略里。
- 仓位公式：`shares = min(max_risk/risk_per_share, max_notional/entry)`，
  整数股向下取整，禁止向上取整扩大风险。
- 所有止盈/止损/熔断阈值必须可配置，且未回测验证前不得声称有效。

## 5. 测试

```bash
# 从仓库根目录运行（推荐）
intraday_scalper/.venv/Scripts/python.exe -m pytest intraday_scalper/tests -q

# 或从本目录运行（pyproject 已配置 pythonpath = [".."]）
.venv/Scripts/python.exe -m pytest tests -q
```

- 测试必须**离线、确定性、可重复**：不联网、不依赖真实时钟、不依赖本地
  历史数据；用 `tests/conftest.py` 的合成数据 fixture。
- 时间相关测试必须注入固定 UTC 时刻（如 `2026-11-06 14:30 UTC` = 09:30 ET）。
- 新增功能必须补测试；禁止删除/弱化既有断言来"让测试通过"。

## 6. 目录职责

| 目录 | 职责 |
|---|---|
| `core/` | 跨模块契约（models / timeutil / config）。改动需经 Lead 评审。 |
| `providers/` | Finnhub WebSocket / REST / 数据质量监控。 |
| `aggregation/` | tick→1m、1m→5m、真实 VWAP、开盘区间。 |
| `features/` | 指标（EMA/RSI/ATR/OR）纯函数，禁止 I/O。 |
| `strategies/` | 三个确定性策略与 registry。 |
| `risk/` | 仓位计算、风控状态机、熔断/冷却/互斥。 |
| `signals/` | 信号引擎与双轨状态机（只提醒，不成交）。 |
| `journal/` | SQLite 人工交易日志与 ENTRY/EXIT/STOP/CANCEL/FLAT 命令。 |
| `dashboard/` | Rich 终端看板与 winsound 声音提醒。 |
| `backtest/` | 事件驱动回测、指标、walk-forward、合成数据。 |
| `data/` | 原始 tick / 聚合 bar 的本地落盘（`var/`，已 gitignore）。 |
| `config/` | `default.yaml`（参数）与 `.env.example`（密钥模板）。 |
| `var/` | 运行时数据、日志、SQLite、回测输出（**不提交**）。 |

## 7. 免责声明

本系统不构成投资建议，不保证盈利。先证明数据可靠，再证明策略有统计优势，
最后才考虑实盘。任何"稳定盈利"表述都是禁止的。
