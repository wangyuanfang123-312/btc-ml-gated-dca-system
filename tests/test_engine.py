import pandas as pd

from btc_dca.engine import run_backtest
from btc_dca.stress import _crash


def config(**overrides):
    values = {
        "initial_equity": 1000.0, "base_order_fraction": 0.15,
        "safety_order_fraction": 0.13, "safety_levels": [-0.04, -0.09, -0.16, -0.26, -0.40],
        "max_cycle_notional_fraction": 0.80, "take_profit_activation": 0.015,
        "trailing_drawdown": 0.005, "hard_equity_floor_fraction": 0.55,
        "emergency_margin_ratio": 0.50, "research_maintenance_margin_rate": 0.01,
        "taker_fee_rate": 0.0005, "slippage_rate": 0.0005,
        "quantity_step": 0.001, "halt_after_risk_exit": True,
    }
    values.update(overrides)
    return values


def prices(closes, lows=None, highs=None):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="5min", tz="UTC")
    close = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": close, "high": highs or close,
                         "low": lows or close, "close": close, "volume": 1.0}, index=idx)


def test_base_and_safety_fill_average_and_exposure_cap():
    bars = prices([100.0, 100.0, 95.0, 95.0], lows=[100.0, 100.0, 95.0, 95.0])
    result = run_backtest(bars, config())
    buys = result.trades.loc[result.trades["side"] == "BUY"]
    assert buys["reason"].tolist() == ["Base", "SO1"]
    assert result.equity["notional"].max() <= 1000 * 0.8
    assert result.cycles.iloc[0]["max_safety_depth"] == 1
    assert result.equity.iloc[-1]["average_price"] > 0
    assert result.summary["max_exposure_fraction_of_cycle_start"] <= 0.80


def test_all_five_safety_layers_and_weighted_average_are_recorded():
    bars = prices([100.0, 96.0, 91.0, 84.0, 74.0, 60.0])
    result = run_backtest(bars, config())
    buys = result.trades.loc[result.trades["side"] == "BUY"]
    assert buys["reason"].tolist() == ["Base", "SO1", "SO2", "SO3", "SO4", "SO5"]
    expected_average = (buys["fill_price"] * buys["qty"]).sum() / buys["qty"].sum()
    assert abs(buys.iloc[-1]["average_price"] - expected_average) < 1e-10
    assert result.cycles.iloc[0]["max_safety_depth"] == 5
    assert result.summary["max_exposure_fraction_of_cycle_start"] <= 0.80


def test_off_blocks_dca_and_on_resumes_unfilled_safety_order():
    bars = prices([100.0] * 48 + [95.0] * 49)
    signal = pd.DataFrame(
        {"p_unsafe": [0.10, 0.40, 0.10], "ml_on": [True, False, True]},
        index=[bars.index[0], bars.index[48], bars.index[96]],
    )
    result = run_backtest(bars, config(), signal=signal)
    buys = result.trades.loc[result.trades["side"] == "BUY"]
    assert buys["reason"].tolist() == ["Base", "SO1"]
    assert buys.iloc[1]["time"] == bars.index[96]


def test_same_bar_multiple_safety_orders_precede_trailing_exit():
    bars = prices([100.0, 100.0], lows=[100.0, 50.0], highs=[100.0, 110.0])
    result = run_backtest(bars, config())
    reasons = result.trades["reason"].dropna().tolist()
    assert reasons[:6] == ["Base", "SO1", "SO2", "SO3", "SO4", "SO5"]
    assert reasons[-1] == "trailing_stop_same_bar"
    assert result.summary["ambiguous_bars"] == 1


def test_equity_floor_closes_and_halts_new_entries():
    idx = pd.date_range("2024-01-01", periods=60, freq="5min", tz="UTC")
    close = [100.0] * 48 + [20.0] * 12
    bars = prices(close)
    bars.index = idx
    result = run_backtest(bars, config(hard_equity_floor_fraction=0.99))
    assert "hard_equity_floor" in result.trades["reason"].tolist()
    assert result.equity["halted"].any()
    assert (result.trades["reason"] == "Base").sum() == 1
    assert not result.trades["reason"].astype(str).str.startswith("SO").any()


def test_trailing_exit_and_funding_are_in_ledger():
    bars = prices([100.0, 100.0, 102.0, 101.0], lows=[100.0, 100.0, 100.0, 100.0],
                  highs=[100.0, 100.0, 102.0, 101.0])
    funding = pd.DataFrame({"funding_rate": [0.001]}, index=[bars.index[1]])
    result = run_backtest(bars, config(), funding=funding)
    assert "FUNDING" in result.trades["side"].tolist()
    assert any(reason.startswith("trailing_stop") for reason in result.trades["reason"].dropna())
    assert result.summary["funding_cost"] > 0


def test_execution_delay_and_injected_missed_order_are_recorded():
    bars = prices([100.0, 100.0, 95.0, 95.0, 95.0])
    delayed = run_backtest(bars, config(), execution_delay_bars=1)
    assert delayed.trades.iloc[0]["time"] == bars.index[1]
    missed = run_backtest(bars, config(), missed_order_index=1)
    assert (missed.trades["side"] == "SKIPPED").any()


def test_crash_stress_changes_one_bar_and_keeps_following_prices():
    bars = prices([100.0, 100.0, 101.0, 102.0])
    mark = bars.copy()
    stressed, stressed_mark = _crash(bars, mark, bars.index[2], -0.10)
    assert stressed.iloc[2]["close"] == 90.0
    assert stressed.iloc[2]["high"] == 100.0
    assert stressed.iloc[3].equals(bars.iloc[3])
    assert stressed_mark.iloc[3].equals(mark.iloc[3])
