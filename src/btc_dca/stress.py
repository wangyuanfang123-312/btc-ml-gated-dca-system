from __future__ import annotations

import pandas as pd

from .engine import BacktestResult, run_backtest
from .walkforward import hysteresis


def _shift_signal(signal: pd.DataFrame, delay_4h_bars: int) -> pd.DataFrame:
    moved = signal.copy()
    moved.index = moved.index + pd.Timedelta(hours=4 * delay_4h_bars)
    moved["ml_on"] = hysteresis(moved["p_unsafe"])
    return moved


def _crash(frame: pd.DataFrame, mark: pd.DataFrame, when: pd.Timestamp,
           drop: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    stressed = frame.copy()
    stressed_mark = mark.copy()
    idx = stressed.index.get_indexer([when])[0]
    if idx <= 0:
        raise ValueError(f"Crash timestamp must have a prior bar: {when}")
    factor = 1.0 + drop
    previous = float(stressed.iloc[idx - 1]["close"])
    previous_mark = float(stressed_mark.iloc[idx - 1]["close"])
    price_cols = ["open", "high", "low", "close"]
    stressed.iloc[idx, stressed.columns.get_indexer(price_cols)] = [
        previous, previous, previous * factor, previous * factor,
    ]
    stressed_mark.iloc[idx, stressed_mark.columns.get_indexer(price_cols)] = [
        previous_mark, previous_mark, previous_mark * factor, previous_mark * factor,
    ]
    return stressed, stressed_mark


def _run_case(name: str, bars: pd.DataFrame, mark: pd.DataFrame, funding: pd.DataFrame,
              signal: pd.DataFrame, config: dict, use_mark_for_risk: bool, **kwargs) -> dict:
    risk_funding = funding if use_mark_for_risk else None
    result = run_backtest(bars, config, signal=signal, mark=mark, funding=risk_funding,
                          use_mark_for_risk=use_mark_for_risk, **kwargs)
    return {"scenario": name, **result.summary}


def run_stress_suite(bars: pd.DataFrame, mark: pd.DataFrame, funding: pd.DataFrame,
                     signal: pd.DataFrame, config: dict, baseline: BacktestResult,
                     use_mark_for_risk: bool = True) -> pd.DataFrame:
    cases = [{"scenario": "baseline" if use_mark_for_risk else "baseline_trade_price_proxy", **baseline.summary}]
    for multiplier in (2, 3):
        cases.append(_run_case(f"taker_fee_x{multiplier}", bars, mark, funding, signal, config, use_mark_for_risk,
                               fee_rate=float(config["taker_fee_rate"]) * multiplier))
    for slippage in (0.001, 0.002, 0.005):
        cases.append(_run_case(f"slippage_{slippage:.3%}", bars, mark, funding, signal, config, use_mark_for_risk,
                               slippage_rate=slippage))
    delayed = _shift_signal(signal, 1)
    cases.append(_run_case("signal_delay_4h", bars, mark, funding, delayed, config, use_mark_for_risk))
    cases.append(_run_case("api_execution_delay_5m", bars, mark, funding, signal, config, use_mark_for_risk,
                            execution_delay_bars=1))
    cases.append(_run_case("missed_third_order_once", bars, mark, funding, signal, config, use_mark_for_risk,
                            missed_order_index=3))

    eligible = [row for row in baseline.cycles.to_dict("records") if row.get("start_time") is not None]
    if eligible:
        longest = max(eligible, key=lambda row: (pd.Timestamp(row["end_time"]) - pd.Timestamp(row["start_time"])).total_seconds())
        crash_time = pd.Timestamp(longest["start_time"]) + (pd.Timestamp(longest["end_time"]) - pd.Timestamp(longest["start_time"])) / 2
    else:
        crash_time = bars.index[len(bars) // 2]
    crash_time = bars.index[bars.index.get_indexer([crash_time], method="nearest")[0]]
    for drop in (-0.10, -0.20):
        shocked_bars, shocked_mark = _crash(bars, mark, crash_time, drop)
        cases.append(_run_case(f"single_5m_gap_{drop:.0%}", shocked_bars, shocked_mark,
                               funding, signal, config, use_mark_for_risk))
    result = pd.DataFrame(cases)
    result["crash_timestamp"] = None
    result.loc[result["scenario"].str.startswith("single_5m_gap_"), "crash_timestamp"] = crash_time.isoformat()
    return result
