from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _fmt(value, pct: bool = False, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    return f"{number:.{digits}%}" if pct else f"{number:,.{digits}f}"


def _write_equity_plot(output: Path, summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    colors = {"V1_oos": "#405d78", "V2_oos": "#a67636", "V1_realistic": "#71918a", "V3_realistic": "#8c5870"}
    for name in ["V1_oos", "V2_oos", "V1_realistic", "V3_realistic"]:
        path = output / f"{name}_equity.parquet"
        if path.exists():
            curve = pd.read_parquet(path)["equity"].resample("1D").last().dropna()
            ax.plot(curve.index, curve, label=name, linewidth=1.25, color=colors[name])
    ax.set_title("BTCUSDT DCA equity by strategy (daily marked value)")
    ax.set_ylabel("Equity (USDT)")
    ax.set_xlabel("UTC date")
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(output / "equity_oos.png", dpi=150)
    plt.close(fig)


def _write_drawdown_plot(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5))
    colors = {"V1_oos": "#405d78", "V2_oos": "#a67636", "V1_realistic": "#71918a", "V3_realistic": "#8c5870"}
    for name in colors:
        path = output / f"{name}_equity.parquet"
        if path.exists():
            series = pd.read_parquet(path)["equity"].resample("1D").last().dropna()
            dd = series / series.cummax() - 1
            ax.plot(dd.index, dd, label=name, linewidth=1.1, color=colors[name])
    ax.set_title("BTCUSDT DCA drawdown from prior equity peak")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("UTC date")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(output / "drawdown_oos.png", dpi=150)
    plt.close(fig)


def _write_model_plot(output: Path, folds: pd.DataFrame) -> None:
    if folds.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for model, color in [("logistic", "#405d78"), ("lightgbm", "#a67636")]:
        part = folds.loc[folds["model"] == model].sort_values("test_start")
        axes[0].plot(part["test_start"], part["test_pr_auc"], marker="o", ms=3,
                     linewidth=1, label=model, color=color)
        axes[1].plot(part["test_start"], part["test_brier"], marker="o", ms=3,
                     linewidth=1, label=model, color=color)
    axes[0].set_title("Unsafe-label precision-recall AUC by walk-forward test fold")
    axes[0].set_ylabel("PR-AUC (higher is better)")
    axes[1].set_title("Unsafe-label Brier score by walk-forward test fold")
    axes[1].set_ylabel("Brier score (lower is better)")
    axes[1].set_xlabel("Test-fold start (UTC)")
    for ax in axes:
        ax.grid(True, color="#dddddd", linewidth=0.6)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "walk_forward_model_metrics.png", dpi=150)
    plt.close(fig)


def _write_stress_plot(output: Path, stress: pd.DataFrame) -> None:
    if stress.empty:
        return
    view = stress.loc[stress["scenario"] != "baseline"].copy()
    view = view.sort_values("max_drawdown")
    fig, ax = plt.subplots(figsize=(12, max(5, len(view) * 0.36)))
    ax.barh(view["scenario"], view["max_drawdown"], color="#617c88")
    ax.set_title("Stress-test maximum drawdown by scenario")
    ax.set_xlabel("Maximum drawdown (%)")
    ax.set_ylabel("Stress scenario")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(True, axis="x", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output / "stress_max_drawdown.png", dpi=150)
    plt.close(fig)


def build_report(output: Path, quality: dict, summary: pd.DataFrame,
                 folds: pd.DataFrame, importance: pd.DataFrame, stress: pd.DataFrame,
                 predictions: pd.DataFrame | None = None,
                 periods: pd.DataFrame | None = None) -> Path:
    _write_equity_plot(output, summary)
    _write_drawdown_plot(output)
    _write_model_plot(output, folds)
    _write_stress_plot(output, stress)

    comparison = summary.set_index("strategy")
    if {"V1_realistic", "V3_realistic"}.issubset(comparison.index):
        realistic_v1 = comparison.loc["V1_realistic"]
        realistic_v3 = comparison.loc["V3_realistic"]
        passed = (
            float(realistic_v3["worst_cycle_loss"]) > float(realistic_v1["worst_cycle_loss"])
            and float(realistic_v3["max_drawdown"]) > float(realistic_v1["max_drawdown"])
            and int(realistic_v3["modeled_liquidations"]) == 0
        )
        status = "通过研究门槛" if passed else "未通过预设研究门槛"
        gate_detail = "V3_realistic 与 V1_realistic 在相同资金费率/标记价格口径下比较。"
    else:
        passed = None
        missing = int(quality.get("oos_missing_mark_rows", 0))
        status = f"V3 未运行：样本外仍有 {missing} 根官方标记价格缺失"
        gate_detail = "按照回测约定，缺少官方标记价格时阻止 V3 和 V1_realistic；本次结果无法判定 ML 是否通过现实口径研究门槛。"
    metric_rows = []
    selected_metrics = ["total_return", "cagr", "max_drawdown", "sharpe", "sortino", "calmar",
                        "profit_factor", "worst_cycle_loss", "max_safety_depth",
                        "max_exposure_fraction_of_initial", "max_exposure_fraction_of_cycle_start",
                        "time_in_market", "ml_off_fraction", "ambiguous_bars", "equity_floor_hits",
                        "risk_exits", "modeled_liquidations", "fees", "slippage_cost", "funding_cost"]
    labels = {"total_return": "Total return", "cagr": "CAGR", "max_drawdown": "Max drawdown",
              "sharpe": "Sharpe", "sortino": "Sortino", "calmar": "Calmar",
              "profit_factor": "Cycle net profit factor", "worst_cycle_loss": "Worst cycle return",
              "max_safety_depth": "Max SO depth", "max_exposure_fraction_of_initial": "Max notional / overall initial equity",
              "max_exposure_fraction_of_cycle_start": "Max cycle notional / cycle start equity",
              "time_in_market": "Time in market", "ambiguous_bars": "Ambiguous 5m bars",
              "equity_floor_hits": "Equity floor hits",
              "ml_off_fraction": "ML OFF share",
              "risk_exits": "Risk exits", "modeled_liquidations": "Modeled liquidations",
              "fees": "Fees (USDT)", "slippage_cost": "Slippage cost (USDT)", "funding_cost": "Funding cost (USDT)"}
    for strategy in summary["strategy"].tolist():
        values = comparison.loc[strategy]
        for metric in selected_metrics:
            value = values.get(metric, np.nan)
            if metric in {"total_return", "cagr", "max_drawdown", "worst_cycle_loss", "time_in_market", "ml_off_fraction"}:
                rendered = _fmt(value, pct=True)
            elif metric in {"max_safety_depth", "ambiguous_bars", "equity_floor_hits", "risk_exits", "modeled_liquidations"}:
                rendered = _fmt(value, digits=0)
            else:
                rendered = _fmt(value)
            metric_rows.append(f"| {strategy} | {labels[metric]} | {rendered} |")

    model_rows = []
    if not folds.empty:
        for model, group in folds.groupby("model"):
            model_rows.append(
                f"| {model} | {len(group)} | {_fmt(group['test_unsafe_recall'].mean(), pct=True)} | "
                f"{_fmt(group['test_unsafe_precision'].mean(), pct=True)} | {_fmt(group['test_pr_auc'].mean())} | "
                f"{_fmt(group['test_roc_auc'].mean())} | {_fmt(group['test_brier'].mean())} |"
            )
    top_features = importance.groupby("feature")["importance"].mean().sort_values(ascending=False).head(10) if not importance.empty else pd.Series(dtype=float)
    feature_lines = "\n".join(f"- `{name}`: mean LightGBM split importance {_fmt(value, digits=1)}" for name, value in top_features.items()) or "- No feature-importance rows were produced."
    stress_rows = []
    for _, row in stress.iterrows():
        stress_rows.append(f"| {row['scenario']} | {_fmt(row.get('total_return'), pct=True)} | "
                           f"{_fmt(row.get('max_drawdown'), pct=True)} | {_fmt(row.get('worst_cycle_loss'), pct=True)} | "
                           f"{_fmt(row.get('modeled_liquidations'), digits=0)} | {_fmt(row.get('risk_exits'), digits=0)} |")
    period_rows = []
    if periods is not None:
        for _, row in periods.iterrows():
            label = f"{str(row['test_start'])[:10]} → {str(row['test_end_exclusive'])[:10]}"
            if bool(row.get("partial_period", False)):
                label += "（部分期）"
            period_rows.append(
                f"| {int(row['fold'])} | {label} | {_fmt(row['V1_oos_return'], pct=True)} | "
                f"{_fmt(row['V2_oos_return'], pct=True)} | {_fmt(row['v2_minus_v1_return'], pct=True)} | "
                f"{_fmt(row['V1_oos_max_drawdown'], pct=True)} | {_fmt(row['V2_oos_max_drawdown'], pct=True)} | "
                f"{_fmt(row['v2_minus_v1_max_drawdown'], pct=True)} | "
                f"{_fmt(row['V1_oos_worst_completed_cycle'], pct=True)} | "
                f"{_fmt(row['V2_oos_worst_completed_cycle'], pct=True)} |"
            )

    quality_lines = [
        f"- OHLCV：{quality['ohlcv']['rows']:,} 根 5 分钟 K 线，{quality['ohlcv']['files']} 个月，"
        f"{quality['ohlcv']['first_time']} 至 {quality['ohlcv']['last_time']}。",
        f"- 标记价格：{quality.get('mark_observed_rows', quality['mark']['rows']):,} 根有效记录；原始归档有 {quality['mark']['invalid_ohlc_rows']} 根缺值。",
        f"- 资金费率：{quality['funding']['rows']:,} 条；费率间隔小时值：{quality['funding']['interval_hours']}。",
        f"- 校验：OHLCV 缺口 {quality['ohlcv']['missing_5m_bars']}、重复 {quality['ohlcv']['duplicate_timestamps']}、异常 OHLC {quality['ohlcv']['invalid_ohlc_rows']}；"
        f"标记价格缺失值 {quality['mark']['invalid_ohlc_rows']} 根，其中样本外 {quality.get('oos_missing_mark_rows', 0)} 根。",
        f"- 日档修复：已核验并补入 {quality.get('mark_daily_repair', {}).get('daily_rows_added', 0)} 根官方标记价格；剩余缺失的时间戳列在 `data_quality.json`。",
        f"- Binance SHA-256 校验：OHLCV {quality['ohlcv']['checksum_checked']} 个月、标记价格 {quality['mark']['checksum_checked']} 个月、资金费率 {quality['funding']['checksum_checked']} 个月。",
    ]
    model_table = "\n".join(model_rows) if model_rows else "| — | 0 | — | — | — | — | — |"
    stress_table = "\n".join(stress_rows) if stress_rows else "| — | — | — | — | — | — |"
    proxy_comparison = ""
    if {"V1_oos", "V2_oos"}.issubset(comparison.index):
        v1 = comparison.loc["V1_oos"]
        v2 = comparison.loc["V2_oos"]
        return_delta = float(v2["total_return"] - v1["total_return"])
        dd_delta = float(v2["max_drawdown"] - v1["max_drawdown"])
        worst_delta = float(v2["worst_cycle_loss"] - v1["worst_cycle_loss"])
        proxy_comparison = (
            f"\n共同样本外代理口径下，V1 收益 {_fmt(v1['total_return'], pct=True)}、最大回撤 "
            f"{_fmt(v1['max_drawdown'], pct=True)}、最差单轮 {_fmt(v1['worst_cycle_loss'], pct=True)}；"
            f"V2 分别为 {_fmt(v2['total_return'], pct=True)}、{_fmt(v2['max_drawdown'], pct=True)}、"
            f"{_fmt(v2['worst_cycle_loss'], pct=True)}。V2 对 V1 的收益差为 {_fmt(return_delta, pct=True)}，"
            f"回撤差为 {_fmt(dd_delta, pct=True)}，最差单轮差为 {_fmt(worst_delta, pct=True)}。"
            "该代理比较没有资金费率和官方标记价格风险计算。"
        )
    if predictions is not None and not predictions.empty:
        model_predictions = predictions.loc[predictions["model"] == "lightgbm"] if "model" in predictions else predictions
        max_probability = float(model_predictions["p_unsafe"].max()) if not model_predictions.empty else np.nan
    else:
        max_probability = np.nan
    off_share = float(comparison.loc["V2_oos", "ml_off_fraction"]) if "V2_oos" in comparison.index else np.nan
    gate_observation = (
        f"\nLightGBM 样本外最大 Unsafe 概率为 {_fmt(max_probability, pct=True)}；"
        f"按 35% 关闭阈值，V2 全部 5 分钟执行时段中的 ML OFF 占比为 {_fmt(off_share, pct=True)}。"
        "概率未达到关闭阈值，开关几乎全程保持 ON，不能把本次差异解释为充分启停过滤的证据。"
        if np.isfinite(max_probability) and np.isfinite(off_share) else ""
    )
    strategy_files = ", ".join(f"`{name}_equity.parquet`" for name in summary["strategy"].tolist())
    trade_cycle_files = ", ".join(f"`{name}_trades.csv` / `{name}_cycles.csv`" for name in summary["strategy"].tolist())
    report = f"""# BTCUSDT 4H ML-Gated DCA 回测结果

## 结论

研究门槛：**{status}**。规则要求现实成本口径下 V3 相对 V1 同时改善最差单轮收益和最大回撤，并且模拟强平次数为零。该判断只适用于本报告覆盖的数据和模拟假设。

比较期：2022-04-01 至 2026-08-31。V1_full 另覆盖全部输入历史。V1/V2 使用相同手续费和滑点；本次缺少样本外官方标记价格，因此 V1_realistic/V3_realistic 未运行。已下载并校验资金费率数据，但未向本次 V1/V2 结果计入资金费率。

{gate_detail}
{proxy_comparison}
{gate_observation}

## 数据质量

{chr(10).join(quality_lines)}

Unsafe 标签正例率：{_fmt(quality['label_positive_rate'], pct=True)}。输入来源与校验结果见 `data_quality.json`。

## 策略对比

| 策略 | 指标 | 数值 |
|---|---|---:|
{chr(10).join(metric_rows)}

![样本外资金曲线](equity_oos.png)

![样本外回撤](drawdown_oos.png)

## 分期样本外策略结果

各期收益与回撤取自同一条连续样本外资金曲线；期初权益使用该期开始前一根 5 分钟 K 线的收盘权益，首期使用 2,000 USDT。策略状态与持仓不在测试期边界重置。最差单轮列出该期结束的已完成 Cycle，跨期未结束的 Cycle 不计入该列。

| Fold | Test window | V1 return | V2 return | V2 − V1 return | V1 max DD | V2 max DD | V2 − V1 DD | V1 worst closed cycle | V2 worst closed cycle |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(period_rows) if period_rows else '| — | — | — | — | — | — | — | — | — | — |'}

完整逐期数字与 Cycle 计数见 `oos_period_comparison.csv`。

## 模型逐期结果

| 模型 | 测试折数 | Unsafe Recall (>35%) | Unsafe Precision (>35%) | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|
{model_table}

![逐期分类指标](walk_forward_model_metrics.png)

前十个平均 LightGBM 分裂重要度：

{feature_lines}

## 压力测试

压力情景在最长样本外持仓 Cycle 中点，将单根 5 分钟 K 线改为从前收盘价下跌 10% 或 20% 并收于低点，后续恢复原行情；延迟、漏单和交易成本按场景名执行。若 V3 被阻止，本表基于 V2 与成交价风险代理，不能替代标记价格风控压力测试。

| 场景 | Total return | Max drawdown | Worst cycle return | 模拟强平 | 风控退出 |
|---|---:|---:|---:|---:|---:|
{stress_table}

![压力场景最大回撤](stress_max_drawdown.png)

## 口径与限制

- 只用已收盘 4H OHLCV 生成预测；订单从下一根 5 分钟 K 线开始执行。标签为后续 42 根 4H 最低价相对当前收盘价下跌至少 12%。测试集不参与模型选择、概率校准或阈值设置。
- 初始权益 2,000 USDT；每边 taker 费率 0.05%、滑点 0.05%；交易所杠杆设置 3x。资金费率归档已通过校验，但因官方标记价格缺失，现实口径策略未运行；V1/V2 与其压力测试不含资金费率，`funding_cost=0`。不得将这些结果视为全成本结果。
- 历史账户专属维持保证金档位无法从公开数据恢复。报告以 1% 研究假设估算维持保证金比率；模拟强平为零不代表所有实际账户或极端成交下不会强平。
- 5 分钟 K 线无法还原 K 线内部真实成交先后。报告对同根 K 线多层 SO 触发和止盈碰撞使用固定保守顺序，并保存相关标记。
- 回测结果是研究证据，不构成实盘收益或安全保证。TradingView 原策略 V0 未提供，因此本报告不声称复现 V0。

## 可复查文件

- `strategy_comparison.csv`：全周期与共同样本外策略指标。
- {strategy_files}：逐 5 分钟权益与持仓。
- {trade_cycle_files}：逐笔订单/费用/盈亏与单轮结果。
- `oos_predictions.parquet`、`walk_forward_metrics.csv`、`lightgbm_feature_importance.csv`：逐期预测、分类表现及特征重要度。
- `oos_period_comparison.csv`：连续样本外曲线按 Walk-Forward 测试窗口的收益、回撤与已完成 Cycle 归因。
- `stress_results.csv`：各压力情景结果。
- `features_4h.parquet`：特征、前瞻标签及 4H 行情。
"""
    path = output / "RESULTS.md"
    path.write_text(report, encoding="utf-8")
    return path

