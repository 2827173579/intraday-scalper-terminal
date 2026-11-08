"""Build a self-contained deliverable bundle for Google Drive / external review.

The bundle is written under ``intraday_scalper/var/gdrive_export/<name>/`` and
zipped next to it.  It contains the audit reports, real-data backtest ledgers,
V0.2 research outputs and a source zip, plus an index and a ready-to-paste
Gemini review prompt.  ``var/`` is git-ignored, so nothing is committed.

Usage::

    python -m intraday_scalper.tools.export_deliverables
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, List

from intraday_scalper.core.config import PACKAGE_ROOT

EXPORT_ROOT = PACKAGE_ROOT / "var" / "gdrive_export"
BUNDLE_NAME = f"intraday_scalper_audit_{date.today().isoformat()}"
EXCLUDE_DIRS = {".venv", "__pycache__", ".pytest_cache", "var", ".git", ".secrets"}
EXCLUDE_FILES = {".env"}

INDEX = """# intraday_scalper · 审计与回测交付包（{bundle}）

> 生成：{generated} ｜ 数据源：Alpaca Markets **IEX 单一交易所** 1m bars（2024-01-02 ~ 2026-11-05，467,120 根）
> ⚠️ 本包用于审计/复核，**不是投资建议**；IEX 成交量仅占全市场一小部分，VWAP 为 IEX-trade VWAP，不是 consolidated tape。

## 这个包是什么

独立子项目 `intraday_scalper`：SOXL / SOXS 分钟级日内信号系统（只做分析/信号/提醒，
绝不自动下单）。本包记录其现有代码审计、真实数据回测、V0.1→V0.2 结构迭代的**全部证据链**。

## 文件地图

| 路径 | 内容 |
|---|---|
| `reports/AUDIT.md` (+.txt) | 旧仓库 `us-market-radar` 现有代码只读审计（599 行，含 file:line 证据与 Gemini 方案差异表 20 条） |
| `reports/data_access_report.md` (+.txt) | Finnhub 免费档 39 组权限实测（/quote 仅昨收、candle 403、WS 无美股成交） |
| `reports/alpaca_probe_result.md` (+.txt) | Alpaca 凭证验证（旧密钥 401 → 新密钥 200、WS auth+subscribe、IEX 口径） |
| `reports/verification_report.md` (+.txt) | 总验证报告：测试 535 passed、Alpaca 接入、真实数据回测、**§9 归因四表**、**§10 V0.2 对比**、**§11 V0.3 消融** |
| `reports/attribution.md` | V0.1 单页归因报告（四表，与 §9 同源） |
| `reports/README.md` / `reports/CLAUDE.md` | 项目说明与开发守则 |
| `backtest_v01/real_iex_report.json` | V0.1 全周期 + 29 折 WF 机读报告 |
| `backtest_v01/real_iex_trades.csv` | V0.1 **2,687 笔逐笔流水**（信号价/成交价/止损/止盈/R/MAE-MFE/出场原因/滑点） |
| `backtest_v01/real_iex_equity_curve.csv` | V0.1 逐日已实现权益 |
| `backtest_v01/attribution.json` | 归因四表结构化数据 |
| `backtest_v02/v02_report.json` | V0.2 全周期/分段/29 折 WF + V0.1 baseline + delta |
| `backtest_v02/v02_oos.json` | V0.2 29 折逐折样本外指标 |
| `backtest_v02/v02_trades.csv` | V0.2 **1,206 笔逐笔流水** |
| `backtest_v03/v03_report.json` | V0.3 五项累计消融（全周期/分段/29 折 WF）+ V0.2 自洽校验 |
| `backtest_v03/v03_trades.csv` | V0.3 最终版逐笔流水（含保本盾 `breakeven_armed` 标记） |
| `backtest_v03/v03_oos.json` | V0.3 29 折逐折样本外 |
| `backtest_v03/v03_ablation.md` | V0.3 消融表（adjusted 数据） |
| `backtest_v03/v03_raw_ablation.md` | V0.3 **raw 未复权**交叉验证（最终口径，含 16,631 根 SOXS 覆盖差异） |
| `backtest_v03/v03_raw_crosscheck.json` | raw vs adjusted 五步消融机读对比 |
| `backtest_v03/v03_sensitivity.md` | V0.3 结构敏感性 9 格网格（raw，预先声明、全部报告） |
| `backtest_raw/raw_canonical_comparison.md` | **raw canonical 基线：V0.1 / V0.2 / V0.3 统一对比** |
| `reports/STRATEGY_ARCHIVE.md` | **策略研究归档说明（负样本外证据、保留资产、重启条件）** |
| `daily_research/OOS_PROTOCOL.md` | 日线/波段雷达的**预注册 OOS 评估协议**（kill criteria 先写后跑） |
| `daily_research/DATA_INVENTORY.md` | 日线面板/旧产物的只读数据清单与已知偏差 |
| `daily_research/CLOSED.md` | **日线 OOS 评估正式封档：只采信两条结论、不买 PIT 数据、不宣称可实盘** |
| `daily_research/results/daily_oos_report.json` | **日线 15 折 OOS 机读结果（D3）** |
| `daily_research/results/VERDICT.md` | **D4 最终裁决（4/5 通过，回撤线未过 → FAIL）** |
| `daily_research/results/folds.csv` / `oos_daily.csv` | 15 折明细 + 945 天逐日组合记录 |
| `daily_research/results/features_frozen.json` | 23 维特征/标签定义 + 特征矩阵哈希 |
| `daily_research/results/D5_KILL_CRITERIA.md` | **D4 vs D5 五条生死线对比（D5 5/5 全过）** |
| `daily_research/results/d5_risk_overlay_report.json` | D5 风控护栏机读报告（ATR 配权 + 暴雨护栏） |
| `backtest_raw/raw_canonical_report.json` | raw canonical 基线机读报告 |
| `code/intraday_scalper_source.zip` | 项目源码（不含虚拟环境/数据/密钥） |
| `02_GEMINI_REVIEW_PROMPT.md` | 可直接粘贴给 Gemini 的复核提示词 |

## 核心数字（供核对）

- 数据：714 个交易日，SOXL 254,479 根 + SOXS 212,641 根 = 467,120 根常规时段 1m bar；0 重复/0 空值/0 OHLC 异常。
- V0.1 全周期：2,687 笔 / 胜率 34.72% / PF 0.909 / 净 **-$1,734.06** / 平均 R -0.083 / 最大回撤 $2,265.07。
- V0.1 WF 样本外：2,341 笔 / 34.05% / PF 0.885 / 净 -$1,956.23 / 9/29 折盈利。
- V0.2 全周期：1,206 笔 / 38.89% / PF 0.967 / 净 **-$385.40** / 平均 R -0.011 / 最大回撤 $1,417.16。
- V0.2 WF 样本外：1,035 笔 / 37.58% / PF 0.918 / 净 -$852.52 / 11/29 折盈利。
- V0.3 全周期（碎股+移除ORB+14:00截止+1.2R保本盾）：945 笔 / 43.81% / PF 1.010 / 净 **+$90.25** / 最大回撤 $509.30。
- V0.3 WF 样本外（adjusted）：805 笔 / 44.10% / PF 0.994 / 净 -$45.75 / 16/29 折盈利。
- ⚠️ **raw canonical（最终口径，见 reports/raw_canonical_comparison.md 与 docs §13）**：
  V0.1 **-$1,184.41**（PF 0.952）→ V0.2 **-$736.64**（0.957）→ V0.3 **-$214.76**（0.977）；
  V0.3 WF 样本外 **-$234.92**（PF 0.970，16/29 盈利折）；结构敏感性 9 格全负（全周期 -$110~-$298，WF -$151~-$294）。
  adjusted 数据上的 V0.3 +$90.25 已被证明是 16,631 根 SOXS bar 覆盖缺失 artifact。
  **结论：结构改进方向真实，但没有任何版本具备正期望，不可实盘。**
- 全项目测试：**535 passed / 0 failed**（新增 44 个 V0.2 离线用例）。
- 既有生产代码 0 修改；未 commit / 未 push。

## 日线/波段 OOS 结果（2026-11-07，预注册 15 折）

日线 23 维模型 + Top-10 组合：OOS 945 天（2022-11 ~ 2026-08），策略 +3,450.03% vs QQQ +171.13%，5 条 Kill Criteria 通过 4 条（收益/CI、PF 1.30、13/15 折、IC t=4.20），**唯一失败：MaxDD 38.03% > 1.5×QQQ（36.33%）→ 判定 FAIL，不可实盘**。注意：35 标的是事后挑选的 AI 赢家（等权基准同期 +1,259%），收益主要来自选择偏差。

**D5 风控附加实验（2026-11-07）**：同一 15 折、同一模型，加入 ATR 逆波动率配权 + 宏观暴雨护栏后，MaxDD 38.03% → **30.47%**（≤36.33% 上限），PF 1.2885，11/15 折正超额，**5/5 条生死线全部通过**；代价是总收益 +3,450% → +1,214%、日均超额 +0.319% → +0.189%。回撤下降主要来自暴雨护栏（仅 ATR 配权为 37.46%，仍不过）。

**日线 OOS 评估已于 2026-11-07 正式封档**：只采信两点——(A) 23 维 LightGBM 有显著但微弱的横截面排序能力（Rank IC 0.0309, t=4.20）；(B) 宏观双均线暴雨护栏能显著削减尾部回撤（38.03%→31.25%/30.47%，仅触发 9.2% 天数）。不买 PIT 退市数据库、不把回测收益当实盘承诺、不宣称策略可实盘。详见 `daily_research/CLOSED.md`。

## 项目状态

**策略研究已归档（2026-11-06）**：raw canonical 下 V0.1/V0.2/V0.3 全周期净盈亏为 -$1,184.41 / -$736.64 / -$214.76，V0.3 WF 样本外 -$234.92，结构敏感性 9 格全负；没有任何版本具备正期望，不可实盘。本包保留完整证据链与可复用基础设施；日线/波段方向已单独预注册 OOS 协议（见 daily_research/）。

## 希望复核的问题（建议顺序）

1. 审计结论是否与仓库现状一致（尤其 main 每 ~90 秒自动 commit、无 candle/WS、无 Firstrade 自动下单、旧回测不可复现）？
2. 归因四表（§9）与逐笔 CSV 是否自洽：2,687 笔、净 -$1,734.06、止损 1,673 笔、1.2R 保本出场 0 笔？
3. V0.1→V0.2 的降噪是否被数字支持；V0.2 仍不盈利、2024 in-sample 正收益未在 2025–2026 holdout 复现；
4. 是否存在未来函数/数据泄漏：5m 只用已完成 K 线、信号下一 5m 开盘成交、同 1m bar TP/SL 碰撞按止损、WF 折间不共享状态；
5. IEX 口径是否被如实披露（成交量/VWAP 非全市场；SOXS 2024 因 $1,200 名义上限 + 高价导致 0 笔成交）。
"""

GEMINI_PROMPT = """请审阅这个 Google Drive 文件夹中的 `intraday_scalper` 审计与回测交付包，扮演一位严格的量化研究/代码审计 Reviewer。

背景（来自文件，请以文件为准）：
- 这是一个 SOXL/SOXS 分钟级日内信号系统（只提醒、人工下单，绝不自动交易）。
- 数据：Alpaca Markets **IEX 单交易所** 1m bars（2024-01-02~2026-11-05，467,120 根）；不是全市场 consolidated tape。
- V0.1（1m 驱动）真实数据：2,687 笔 / 胜率 34.72% / PF 0.909 / 净 -$1,734.06 / 最大回撤 $2,265.07；29 折 WF 样本外 2,341 笔 / PF 0.885 / 净 -$1,956.23。
- V0.2（5m 闭合 Bar + 镜像共振 + IEX 相对口径，未调参）：1,206 笔 / 38.89% / PF 0.967 / 净 -$385.40；WF 样本外 1,035 笔 / PF 0.918 / 净 -$852.52。
- 项目测试 535 passed。

请重点检查并逐条给出「证据 + 判断（通过/存疑/错误）」：
1. `reports/verification_report.md` §9 归因四表与 `backtest_v01/real_iex_trades.csv` 是否自洽；ORB_5M×SOXL、时段 13:30-15:30、止损 1,673 笔、1.2R 保本出场 0 笔等结论是否成立。
2. §10 V0.2 结构改动是否真的只改结构、没有偷调参数；V0.1/V0.2 对比数字是否一致；2024 in-sample 正收益与 2025+ holdout 负收益是否被如实披露。
3. 是否存在未来函数/数据泄漏：5m 只用已完成 K 线？信号是否在下一根 5m 开盘成交？同一根 1m bar 同时 TP/SL 是否按止损保守处理？WF 每折是否使用全新风控且无跨折状态泄漏？
4. 审计报告 `reports/AUDIT.md` 对旧仓库的判断（main 自动 commit、无 candle/WS、无 Firstrade 自动下单、旧回测不可复现）是否有证据支撑。
5. 任何被过度宣称、口径误导、样本不足或应当标注而未标注的地方；以及如果要让 V0.2 继续迭代，最应该优先验证的 3 个结构性假设。
6. V0.3 五项累计消融（`backtest_v03/v03_ablation.md`、`v03_report.json`）：碎股、移除 ORB_15M_5M、14:00 ET 新开仓截止、+1.2R 保本盾各自贡献多少；Step0 是否严格复现 V0.2 基线。
7. **raw（未复权）交叉验证**（`backtest_v03/v03_raw_ablation.md`、`v03_raw_crosscheck.json`）：adjusted 数据的 16,631 根 SOXS 覆盖缺失是否被如实披露；以 raw 为准时 V0.3 是否仍不盈利；碎股修复是否被过度归因。
8. **raw canonical 基线**（`backtest_raw/raw_canonical_comparison.md`）：V0.1/V0.2/V0.3 在 raw 上是否统一口径、版本间是否单调改善、是否仍全部为负；**V0.3 结构敏感性 9 格网格**（`backtest_v03/v03_sensitivity.md`）是否全部为负、是否存在被隐藏的择优。
9. **归档与下一步**（`reports/STRATEGY_ARCHIVE.md`、`daily_research/OOS_PROTOCOL.md`）：策略归档结论是否与证据一致；日线 OOS 协议的 15 折/purge/embargo/kill criteria 是否足够严格、是否披露了幸存者偏差与不平衡面板；是否存在"换个方向继续调参"的隐含冲动。
10. **日线 OOS 裁决**（`reports/daily_oos_verdict.md`、`daily_research/results/`）：4/5 通过但回撤超标被判 FAIL 是否合理；收益是否主要来自事后挑选的 AI 赢家宇宙（等权基准 +1,259%）而非模型；是否存在把"回测赚钱"误读成"实盘能赚钱"的表述。
11. **D5 风控护栏**（`reports/D5_KILL_CRITERIA.md`）：ATR 逆波动率配权 + 宏观暴雨护栏（QQQM/SMH 同时 < -3% 时降到 30% 仓位）是否属于合法的结构性风控（不是调参）；D5 5/5 通过是否只说明"回撤门槛可被风控满足"，而幸存者偏差仍未解决。
12. **封档说明**（`reports/daily_research_CLOSED.md`）：只采信 Rank IC 与暴雨护栏两条结论、拒绝把回测收益当实盘承诺、不买 PIT 数据——这一封档决定是否克制、是否存在遗漏的未披露风险。

请用中文输出一份结构化 Review（不要只复述数字），最后给一个明确结论：这套系统当前是否具备实盘条件？如果不具备，拦住实盘的关键证据是哪三条？
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"missing deliverable source: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _txt_copy(md_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    md_path.with_suffix(".txt").write_text(text, encoding="utf-8")


def _iter_source_files(package_root: Path) -> Iterable[Path]:
    for path in sorted(package_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(package_root)
        if any(part in EXCLUDE_DIRS for part in relative.parts):
            continue
        if path.name in EXCLUDE_FILES:
            continue
        if path.suffix == ".parquet":
            continue
        yield path


def build_source_zip(package_root: Path, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in _iter_source_files(package_root):
            relative = path.relative_to(package_root.parent)
            archive.write(path, arcname=str(Path("intraday_scalper") / relative))
            count += 1
    return count


def build_bundle() -> Path:
    package = PACKAGE_ROOT
    bundle = EXPORT_ROOT / BUNDLE_NAME
    if bundle.exists():
        shutil.rmtree(bundle)
    reports = bundle / "reports"
    v01 = bundle / "backtest_v01"
    v02 = bundle / "backtest_v02"
    v03 = bundle / "backtest_v03"
    raw = bundle / "backtest_raw"
    daily = bundle / "daily_research"
    daily_results = daily / "results"
    code = bundle / "code"
    for directory in (bundle, reports, v01, v02, v03, raw, daily, daily_results, code):
        directory.mkdir(parents=True, exist_ok=True)

    copy_map = {
        package / "AUDIT.md": reports / "AUDIT.md",
        package / "docs" / "data_access_report.md": reports / "data_access_report.md",
        package / "docs" / "alpaca_probe_result.md": reports / "alpaca_probe_result.md",
        package / "docs" / "verification_report.md": reports / "verification_report.md",
        package / "var" / "backtest" / "attribution.md": reports / "attribution.md",
        package / "README.md": reports / "README.md",
        package / "CLAUDE.md": reports / "CLAUDE.md",
        package / "var" / "backtest" / "real_iex_report.json": v01 / "real_iex_report.json",
        package / "var" / "backtest" / "real_iex_equity_curve.csv": v01 / "real_iex_equity_curve.csv",
        package / "var" / "backtest" / "real_iex_trades.csv": v01 / "real_iex_trades.csv",
        package / "var" / "backtest" / "attribution.json": v01 / "attribution.json",
        package / "var" / "backtest" / "v02_report.json": v02 / "v02_report.json",
        package / "var" / "backtest" / "v02_oos.json": v02 / "v02_oos.json",
        package / "var" / "backtest" / "v02_trades.csv": v02 / "v02_trades.csv",
        package / "var" / "backtest" / "v03_report.json": v03 / "v03_report.json",
        package / "var" / "backtest" / "v03_ablation.md": v03 / "v03_ablation.md",
        package / "var" / "backtest" / "v03_trades.csv": v03 / "v03_trades.csv",
        package / "var" / "backtest" / "v03_oos.json": v03 / "v03_oos.json",
        package / "var" / "backtest" / "v03_raw_crosscheck.json": v03 / "v03_raw_crosscheck.json",
        package / "var" / "backtest" / "v03_raw_ablation.md": v03 / "v03_raw_ablation.md",
        package / "var" / "backtest" / "v03_sensitivity.json": v03 / "v03_sensitivity.json",
        package / "var" / "backtest" / "v03_sensitivity.md": v03 / "v03_sensitivity.md",
        package / "var" / "backtest" / "raw_canonical_report.json": raw / "raw_canonical_report.json",
        package / "var" / "backtest" / "raw_canonical_comparison.md": raw / "raw_canonical_comparison.md",
        package / "docs" / "STRATEGY_ARCHIVE.md": reports / "STRATEGY_ARCHIVE.md",
        package.parent / "daily_research" / "OOS_PROTOCOL.md": daily / "OOS_PROTOCOL.md",
        package.parent / "daily_research" / "DATA_INVENTORY.md": daily / "DATA_INVENTORY.md",
        package.parent / "daily_research" / "CLOSED.md": daily / "CLOSED.md",
        package.parent / "daily_research" / "results" / "VERDICT.md": daily_results / "VERDICT.md",
        package.parent / "daily_research" / "results" / "daily_oos_summary.md": daily_results / "daily_oos_summary.md",
        package.parent / "daily_research" / "results" / "daily_oos_report.json": daily_results / "daily_oos_report.json",
        package.parent / "daily_research" / "results" / "features_frozen.json": daily_results / "features_frozen.json",
        package.parent / "daily_research" / "results" / "DEVIATIONS.md": daily_results / "DEVIATIONS.md",
        package.parent / "daily_research" / "results" / "folds.csv": daily_results / "folds.csv",
        package.parent / "daily_research" / "results" / "oos_daily.csv": daily_results / "oos_daily.csv",
        package.parent / "daily_research" / "results" / "D5_KILL_CRITERIA.md": daily_results / "D5_KILL_CRITERIA.md",
        package.parent / "daily_research" / "results" / "d5_risk_overlay_summary.md": daily_results / "d5_risk_overlay_summary.md",
        package.parent / "daily_research" / "results" / "d5_risk_overlay_report.json": daily_results / "d5_risk_overlay_report.json",
        package.parent / "daily_research" / "results" / "d5_daily_D4_equal_no_guard.csv": daily_results / "d5_daily_D4_equal_no_guard.csv",
        package.parent / "daily_research" / "results" / "d5_daily_D5_full.csv": daily_results / "d5_daily_D5_full.csv",
    }
    for source, target in copy_map.items():
        if not source.exists():
            continue
        _copy(source, target)
        if target.suffix == ".md":
            _txt_copy(target)
    ablation_v03 = package / "var" / "backtest" / "v03_ablation.md"
    if ablation_v03.exists():
        _copy(ablation_v03, reports / "v03_ablation.md")
        _txt_copy(reports / "v03_ablation.md")
    raw_ablation_v03 = package / "var" / "backtest" / "v03_raw_ablation.md"
    if raw_ablation_v03.exists():
        _copy(raw_ablation_v03, reports / "v03_raw_ablation.md")
        _txt_copy(reports / "v03_raw_ablation.md")
    sensitivity_v03 = package / "var" / "backtest" / "v03_sensitivity.md"
    if sensitivity_v03.exists():
        _copy(sensitivity_v03, reports / "v03_sensitivity.md")
        _txt_copy(reports / "v03_sensitivity.md")
    canonical_raw = package / "var" / "backtest" / "raw_canonical_comparison.md"
    if canonical_raw.exists():
        _copy(canonical_raw, reports / "raw_canonical_comparison.md")
        _txt_copy(reports / "raw_canonical_comparison.md")
    daily_verdict = package.parent / "daily_research" / "results" / "VERDICT.md"
    if daily_verdict.exists():
        _copy(daily_verdict, reports / "daily_oos_verdict.md")
        _txt_copy(reports / "daily_oos_verdict.md")
    daily_summary = package.parent / "daily_research" / "results" / "daily_oos_summary.md"
    if daily_summary.exists():
        _copy(daily_summary, reports / "daily_oos_summary.md")
        _txt_copy(reports / "daily_oos_summary.md")
    d5_criteria = package.parent / "daily_research" / "results" / "D5_KILL_CRITERIA.md"
    if d5_criteria.exists():
        _copy(d5_criteria, reports / "D5_KILL_CRITERIA.md")
        _txt_copy(reports / "D5_KILL_CRITERIA.md")
    d5_summary = package.parent / "daily_research" / "results" / "d5_risk_overlay_summary.md"
    if d5_summary.exists():
        _copy(d5_summary, reports / "d5_risk_overlay_summary.md")
        _txt_copy(reports / "d5_risk_overlay_summary.md")
    daily_closed = package.parent / "daily_research" / "CLOSED.md"
    if daily_closed.exists():
        _copy(daily_closed, reports / "daily_research_CLOSED.md")
        _txt_copy(reports / "daily_research_CLOSED.md")

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    index_text = INDEX.format(bundle=BUNDLE_NAME, generated=generated)
    (bundle / "00_START_HERE.md").write_text(index_text, encoding="utf-8")
    (bundle / "02_GEMINI_REVIEW_PROMPT.md").write_text(GEMINI_PROMPT, encoding="utf-8")

    # Single plain-text review pack: easiest for Gemini / Drive preview to read
    # in one pass (the individual .md/.txt/.csv files remain available too).
    parts = [
        index_text,
        "# ============ 完整验证报告（含 §9 归因、§10 V0.2） ============",
        (reports / "verification_report.md").read_text(encoding="utf-8"),
        "# ============ V0.3 增量消融（独立单页） ============",
        (reports / "v03_ablation.md").read_text(encoding="utf-8")
        if (reports / "v03_ablation.md").exists()
        else "(V0.3 ablation not generated yet)",
        "# ============ V0.3 raw（未复权）交叉验证（最终口径） ============",
        (reports / "v03_raw_ablation.md").read_text(encoding="utf-8")
        if (reports / "v03_raw_ablation.md").exists()
        else "(V0.3 raw cross-check not generated yet)",
        "# ============ raw canonical 基线（最终口径） ============",
        (reports / "raw_canonical_comparison.md").read_text(encoding="utf-8")
        if (reports / "raw_canonical_comparison.md").exists()
        else "(raw canonical comparison not generated yet)",
        "# ============ V0.3 结构敏感性（raw，预先声明网格） ============",
        (reports / "v03_sensitivity.md").read_text(encoding="utf-8")
        if (reports / "v03_sensitivity.md").exists()
        else "(V0.3 sensitivity not generated yet)",
        "# ============ 策略研究归档说明 ============",
        (reports / "STRATEGY_ARCHIVE.md").read_text(encoding="utf-8")
        if (reports / "STRATEGY_ARCHIVE.md").exists()
        else "(archive note missing)",
        "# ============ 日线/波段雷达 OOS 预注册协议 ============",
        (daily / "OOS_PROTOCOL.md").read_text(encoding="utf-8")
        if (daily / "OOS_PROTOCOL.md").exists()
        else "(daily OOS protocol missing)",
        "# ============ 日线 OOS 最终裁决（D4，大白话版） ============",
        (daily / "results" / "VERDICT.md").read_text(encoding="utf-8")
        if (daily / "results" / "VERDICT.md").exists()
        else "(daily verdict missing)",
        "# ============ 日线 OOS 汇总表 ============",
        (daily / "results" / "daily_oos_summary.md").read_text(encoding="utf-8")
        if (daily / "results" / "daily_oos_summary.md").exists()
        else "(daily summary missing)",
        "# ============ D5 风控护栏：D4 vs D5 五条生死线 ============",
        (daily / "results" / "D5_KILL_CRITERIA.md").read_text(encoding="utf-8")
        if (daily / "results" / "D5_KILL_CRITERIA.md").exists()
        else "(D5 criteria missing)",
        "# ============ D5 风控护栏汇总 ============",
        (daily / "results" / "d5_risk_overlay_summary.md").read_text(encoding="utf-8")
        if (daily / "results" / "d5_risk_overlay_summary.md").exists()
        else "(D5 summary missing)",
        "# ============ 日线 OOS 封档说明 ============",
        (daily / "CLOSED.md").read_text(encoding="utf-8")
        if (daily / "CLOSED.md").exists()
        else "(daily closure note missing)",
        "# ============ V0.1 归因四表（独立单页） ============",
        (reports / "attribution.md").read_text(encoding="utf-8"),
        "# ============ 审计报告 ============",
        (reports / "AUDIT.md").read_text(encoding="utf-8"),
    ]
    (reports / "REVIEW_PACK.txt").write_text("\n\n".join(parts), encoding="utf-8")
    source_zip = code / "intraday_scalper_source.zip"
    source_count = build_source_zip(package, source_zip)

    # Verify internal consistency of the copied ledger counts.
    import csv

    with (v01 / "real_iex_trades.csv").open(encoding="utf-8", newline="") as handle:
        v01_rows = sum(1 for _ in csv.reader(handle)) - 1
    with (v02 / "v02_trades.csv").open(encoding="utf-8", newline="") as handle:
        v02_rows = sum(1 for _ in csv.reader(handle)) - 1
    v01_report = json.loads((v01 / "real_iex_report.json").read_text(encoding="utf-8"))
    v02_report = json.loads((v02 / "v02_report.json").read_text(encoding="utf-8"))
    v01_trades = v01_report["full_period"]["metrics"]["total_trades"]
    v02_trades = v02_report["v02_full_period"]["metrics"]["total_trades"]
    v03_report_path = v03 / "v03_report.json"
    v03_trades_path = v03 / "v03_trades.csv"
    v03_rows = 0
    v03_expected = 0
    if v03_report_path.exists() and v03_trades_path.exists():
        with v03_trades_path.open(encoding="utf-8", newline="") as handle:
            v03_rows = sum(1 for _ in csv.reader(handle)) - 1
        v03_rep = json.loads(v03_report_path.read_text(encoding="utf-8"))
        v03_expected = int(v03_rep["steps"][-1]["full"]["metrics"]["total_trades"])
        if v03_rows != v03_expected:
            raise RuntimeError(f"ledger mismatch: v03 {v03_rows} vs {v03_expected}")
    if v01_rows != v01_trades or v02_rows != v02_trades:
        raise RuntimeError(
            f"ledger mismatch: v01 {v01_rows} vs {v01_trades}; v02 {v02_rows} vs {v02_trades}"
        )

    checksum_lines = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksum_lines.append(f"{_sha256(path)}  {path.relative_to(bundle).as_posix()}")
    (bundle / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    zip_path = EXPORT_ROOT / f"{BUNDLE_NAME}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=f"{BUNDLE_NAME}/{path.relative_to(bundle).as_posix()}")

    total_size = sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file())
    print(f"bundle dir : {bundle}")
    print(f"bundle zip : {zip_path} ({zip_path.stat().st_size / 1_048_576:.2f} MB)")
    print(f"files      : {sum(1 for p in bundle.rglob('*') if p.is_file())} (bundle content {total_size / 1_048_576:.2f} MB)")
    print(f"source zip : {source_zip} ({source_count} files)")
    print(f"ledgers    : v01 {v01_rows} / v02 {v02_rows} / v03 {v03_rows or 'n/a'} rows  (match reports)")
    return bundle


def main(argv: List[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    build_bundle()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
