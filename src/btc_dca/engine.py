from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    cycles: pd.DataFrame
    summary: dict


def run_backtest(
    ohlcv: pd.DataFrame,
    config: dict,
    signal: pd.DataFrame | None = None,
    mark: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    fee_rate: float | None = None,
    slippage_rate: float | None = None,
    use_mark_for_risk: bool = False,
    execution_delay_bars: int = 0,
    missed_order_index: int | None = None,
) -> BacktestResult:
    """Long-only USD-M futures DCA simulator with bar-order rules documented in README."""
    if ohlcv.empty:
        raise ValueError("Backtest requires non-empty OHLCV")
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(ohlcv.columns):
        raise ValueError(f"OHLCV missing columns: {required - set(ohlcv.columns)}")
    if use_mark_for_risk and (mark is None or mark.empty):
        raise ValueError("Mark-price data is required for mark-based risk simulation")

    fee_rate = float(config["taker_fee_rate"] if fee_rate is None else fee_rate)
    slippage_rate = float(config["slippage_rate"] if slippage_rate is None else slippage_rate)
    start_equity = float(config["initial_equity"])
    base_fraction = float(config["base_order_fraction"])
    so_fraction = float(config["safety_order_fraction"])
    max_notional_fraction = float(config["max_cycle_notional_fraction"])
    qty_step = float(config.get("quantity_step", 0.001))
    so_levels = list(config["safety_levels"])
    tp_activation = float(config["take_profit_activation"])
    trail = float(config["trailing_drawdown"])
    floor_fraction = float(config["hard_equity_floor_fraction"])
    emergency_ratio = float(config["emergency_margin_ratio"])
    maintenance_rate = float(config.get("research_maintenance_margin_rate", 0.01))
    gated = signal is not None

    frame = ohlcv.copy()
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("OHLCV index must be timezone-aware UTC timestamps")
    frame = frame.sort_index()
    if gated:
        sig = signal.sort_index()
        p_series = sig["p_unsafe"].reindex(frame.index, method="ffill")
        on_series = sig["ml_on"].astype("boolean").reindex(frame.index, method="ffill").fillna(False)
        p_values = p_series.to_numpy(dtype=float, na_value=np.nan)
        on_values = on_series.to_numpy(dtype=bool)
    else:
        p_values = np.full(len(frame), np.nan)
        on_values = np.ones(len(frame), dtype=bool)
    if use_mark_for_risk:
        mark_aligned = mark.reindex(frame.index)
        if mark_aligned[["open", "high", "low", "close"]].isna().any().any():
            raise ValueError("Mark-price bars are missing or misaligned with execution bars")
        mark_open = mark_aligned["open"].to_numpy(dtype=float)
        mark_high = mark_aligned["high"].to_numpy(dtype=float)
        mark_low = mark_aligned["low"].to_numpy(dtype=float)
        mark_close = mark_aligned["close"].to_numpy(dtype=float)
    else:
        mark_open = frame["open"].to_numpy(dtype=float)
        mark_high = frame["high"].to_numpy(dtype=float)
        mark_low = frame["low"].to_numpy(dtype=float)
        mark_close = frame["close"].to_numpy(dtype=float)

    opens = frame["open"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    timestamps = frame.index
    funding_events: dict[pd.Timestamp, float] = {}
    if funding is not None and not funding.empty:
        funding_events = {pd.Timestamp(t): float(r) for t, r in funding["funding_rate"].items()}

    balance = start_equity
    qty = 0.0
    avg_price = 0.0
    cycle_id = 0
    cycle_start_equity = np.nan
    cycle_start_time = None
    cycle_start_price = np.nan
    filled_so: set[int] = set()
    pending: dict[int, tuple[int, str, float, int]] = {}
    # pending key is order number; value is (due bar, reason, target, layer index).
    pending_base: int | None = None
    trail_active = False
    trail_peak = 0.0
    halted = False
    order_number = 0
    missed_orders = 0
    modeled_liquidations = 0
    risk_exits = 0
    equity_rows: list[tuple] = []
    trades: list[dict] = []
    cycles: list[dict] = []
    cycle_fees = cycle_slippage = cycle_funding = cycle_gross = 0.0
    cycle_max_depth = 0
    cycle_max_notional = 0.0
    cycle_ambiguous = 0
    ambiguous_bars = 0
    ambiguous_timestamps: set[pd.Timestamp] = set()
    fee_total = slip_total = funding_total = gross_total = 0.0

    def is_boundary(ts: pd.Timestamp) -> bool:
        return ts.minute == 0 and ts.second == 0 and ts.hour % 4 == 0

    def finish_cycle(ts: pd.Timestamp, reason: str, equity_value: float) -> None:
        nonlocal cycle_fees, cycle_slippage, cycle_funding, cycle_gross, cycle_id
        nonlocal cycle_max_depth, cycle_max_notional, cycle_ambiguous
        if cycle_id == 0 or cycle_start_time is None:
            return
        cycles.append({
            "cycle_id": cycle_id, "start_time": cycle_start_time, "end_time": ts,
            "start_price": cycle_start_price, "start_equity": cycle_start_equity,
            "end_equity": equity_value,
            "net_pnl": equity_value - cycle_start_equity,
            "net_return": equity_value / cycle_start_equity - 1 if cycle_start_equity else np.nan,
            "exit_reason": reason, "max_safety_depth": cycle_max_depth,
            "max_notional": cycle_max_notional, "fees": cycle_fees,
            "slippage_cost": cycle_slippage, "funding_cost": cycle_funding,
            "gross_realized_pnl": cycle_gross, "ambiguous_bars": cycle_ambiguous,
        })
        cycle_fees = cycle_slippage = cycle_funding = cycle_gross = 0.0
        cycle_max_depth = 0
        cycle_max_notional = 0.0
        cycle_ambiguous = 0

    def buy(i: int, reason: str, reference: float, target: float, order_no: int) -> bool:
        nonlocal qty, avg_price, balance, fee_total, slip_total, cycle_fees, cycle_slippage
        nonlocal cycle_start_equity, cycle_start_time, cycle_start_price, cycle_id
        nonlocal trail_active, trail_peak, cycle_max_notional, cycle_max_depth
        ts = timestamps[i]
        if missed_order_index is not None and order_no == missed_order_index:
            if reason.startswith("SO"):
                filled_so.add(int(reason[2:]))
            trades.append({"time": ts, "cycle_id": cycle_id, "side": "SKIPPED", "reason": "injected_missed_order",
                           "reference_price": reference, "fill_price": np.nan, "qty": 0.0,
                           "notional": 0.0, "fee": 0.0, "slippage_cost": 0.0,
                           "p_unsafe": p_values[i], "ml_on": bool(on_values[i])})
            return False
        if qty <= 0:
            cycle_id += 1
            filled_so.clear()
            cycle_start_equity = balance
            cycle_start_time = ts
            cycle_start_price = reference * (1 + slippage_rate)
            trail_active = False
            trail_peak = 0.0
        fill = reference * (1 + slippage_rate)
        max_after = cycle_start_equity * max_notional_fraction
        remaining = max_after - qty * fill
        notional = min(float(target), remaining)
        candidate = math.floor((notional / fill) / qty_step + 1e-9) * qty_step if notional > 0 else 0.0
        if candidate <= 0:
            return False
        actual_notional = candidate * fill
        old_qty = qty
        qty += candidate
        avg_price = (avg_price * old_qty + fill * candidate) / qty
        fee = actual_notional * fee_rate
        slip = candidate * max(0.0, fill - reference)
        balance -= fee
        fee_total += fee
        slip_total += slip
        cycle_fees += fee
        cycle_slippage += slip
        if reason.startswith("SO"):
            layer = int(reason[2:])
            filled_so.add(layer)
            cycle_max_depth = max(cycle_max_depth, layer)
        cycle_max_notional = max(cycle_max_notional, qty * fill)
        trades.append({
            "time": ts, "cycle_id": cycle_id, "side": "BUY", "reason": reason,
            "reference_price": reference, "fill_price": fill, "qty": candidate,
            "position_qty": qty, "average_price": avg_price, "notional": actual_notional,
            "fee": fee, "slippage_cost": slip, "p_unsafe": p_values[i], "ml_on": bool(on_values[i]),
        })
        return True

    def sell(i: int, reference: float, reason: str) -> None:
        nonlocal qty, avg_price, balance, fee_total, slip_total, gross_total
        nonlocal cycle_fees, cycle_slippage, cycle_gross, trail_active, trail_peak
        ts = timestamps[i]
        if qty <= 0:
            return
        fill = reference * (1 - slippage_rate)
        notional = qty * fill
        gross = qty * (fill - avg_price)
        fee = notional * fee_rate
        slip = qty * max(0.0, reference - fill)
        balance += gross - fee
        gross_total += gross
        fee_total += fee
        slip_total += slip
        cycle_gross += gross
        cycle_fees += fee
        cycle_slippage += slip
        trades.append({
            "time": ts, "cycle_id": cycle_id, "side": "SELL", "reason": reason,
            "reference_price": reference, "fill_price": fill, "qty": qty,
            "position_qty": 0.0, "average_price": avg_price, "notional": notional,
            "gross_pnl": gross, "fee": fee, "slippage_cost": slip,
            "p_unsafe": p_values[i], "ml_on": bool(on_values[i]),
        })
        final_equity = balance
        finish_cycle(ts, reason, final_equity)
        qty = 0.0
        avg_price = 0.0
        trail_active = False
        trail_peak = 0.0

    def risk_reason(i: int) -> str | None:
        if qty <= 0:
            return None
        price = mark_low[i]
        low_equity = balance + qty * (price - avg_price)
        ratio = qty * price * maintenance_rate / low_equity if low_equity > 0 else math.inf
        if ratio >= 1.0:
            return "modeled_liquidation"
        if ratio >= emergency_ratio:
            return "emergency_margin"
        if low_equity <= cycle_start_equity * floor_fraction:
            return "hard_equity_floor"
        return None

    def flag_ambiguous(ts: pd.Timestamp) -> None:
        nonlocal ambiguous_bars, cycle_ambiguous
        if ts not in ambiguous_timestamps:
            ambiguous_timestamps.add(ts)
            ambiguous_bars += 1
            if cycle_id > 0 and qty > 0:
                cycle_ambiguous += 1

    def touched_safety_orders(i: int) -> list[int]:
        pending_layers = {value[3] for value in pending.values()}
        return [layer for layer, level in enumerate(so_levels, start=1)
                if layer not in filled_so and layer not in pending_layers
                and lows[i] <= cycle_start_price * (1 + float(level))]

    n = len(frame)
    for i, ts in enumerate(timestamps):
        if i and (ts - timestamps[i - 1]) != pd.Timedelta(minutes=5):
            raise ValueError(f"Execution gap at {ts}; refusing to simulate across missing 5m bars")
        exited_this_bar = False
        safety_filled_this_bar = False
        current_on = bool(on_values[i]) if gated else True

        # Funding is charged at its scheduled bar open on the position carried into that timestamp.
        rate = funding_events.get(ts)
        if qty > 0 and rate is not None:
            event_notional = qty * mark_open[i]
            charge = event_notional * rate
            balance -= charge
            funding_total += charge
            cycle_funding += charge
            trades.append({"time": ts, "cycle_id": cycle_id, "side": "FUNDING", "reason": "scheduled_funding",
                           "reference_price": mark_open[i], "fill_price": mark_open[i], "qty": qty,
                           "notional": event_notional, "funding_rate": rate, "funding_cost": charge,
                           "fee": 0.0, "slippage_cost": 0.0, "p_unsafe": p_values[i], "ml_on": current_on})

        # A new Base order is placed at the first 5m open after an eligible 4h decision.
        if not halted and qty <= 0 and pending_base is None and is_boundary(ts) and current_on:
            order_number += 1
            if execution_delay_bars > 0:
                pending_base = order_number
                pending[order_number] = (i + execution_delay_bars, "Base", balance * base_fraction, 0)
            else:
                buy(i, "Base", opens[i], balance * base_fraction, order_number)
        intrabar_notional = qty * mark_high[i] if qty > 0 else 0.0
        if qty > 0:
            cycle_max_notional = max(cycle_max_notional, intrabar_notional)

        # Check any delayed API orders before normal intrabar triggers; risk checks still override them.
        initial_risk = risk_reason(i)
        if initial_risk:
            if qty > 0 and touched_safety_orders(i):
                flag_ambiguous(ts)
            risk_exits += 1
            modeled_liquidations += initial_risk == "modeled_liquidation"
            sell(i, min(opens[i], lows[i]), initial_risk)
            exited_this_bar = True
            halted = bool(config.get("halt_after_risk_exit", True))
            pending.clear()
            pending_base = None

        if not exited_this_bar and not halted and pending:
            for order_no, (due_i, reason, target, layer) in list(pending.items()):
                if due_i <= i:
                    if current_on and (layer == 0 or qty > 0):
                        success = buy(i, reason, opens[i], target, order_no)
                        if success and layer > 0:
                            filled_so.add(layer)
                            safety_filled_this_bar = True
                    pending.pop(order_no, None)
                    if reason == "Base":
                        pending_base = None

        # Existing trailing protection is evaluated before lower DCA limits.
        if not exited_this_bar and qty > 0 and trail_active:
            prior_stop = trail_peak * (1 - trail)
            if lows[i] <= prior_stop:
                if touched_safety_orders(i):
                    flag_ambiguous(ts)
                sell(i, min(opens[i], prior_stop), "trailing_stop")
                exited_this_bar = True
        if not exited_this_bar and qty > 0 and not trail_active:
            touched = touched_safety_orders(i)
            if len(touched) > 1:
                flag_ambiguous(ts)
            for layer in touched:
                if not current_on:
                    continue
                order_number += 1
                target_price = cycle_start_price * (1 + float(so_levels[layer - 1]))
                target_notional = cycle_start_equity * so_fraction
                if execution_delay_bars > 0:
                    pending[order_number] = (i + execution_delay_bars, f"SO{layer}", target_notional, layer)
                else:
                    if buy(i, f"SO{layer}", target_price, target_notional, order_number):
                        filled_so.add(layer)
                        safety_filled_this_bar = True
                        intrabar_notional = max(intrabar_notional, qty * mark_high[i])
                        cycle_max_notional = max(cycle_max_notional, qty * mark_high[i])
        if qty > 0:
            intrabar_notional = max(intrabar_notional, qty * mark_high[i])
            cycle_max_notional = max(cycle_max_notional, intrabar_notional)
        if not exited_this_bar and qty > 0:
            post_order_risk = risk_reason(i)
            if post_order_risk:
                if safety_filled_this_bar:
                    flag_ambiguous(ts)
                risk_exits += 1
                modeled_liquidations += post_order_risk == "modeled_liquidation"
                sell(i, min(opens[i], lows[i]), post_order_risk)
                exited_this_bar = True
                halted = bool(config.get("halt_after_risk_exit", True))
                pending.clear()
                pending_base = None
        if not exited_this_bar and qty > 0:
            if trail_active:
                trail_peak = max(trail_peak, highs[i])
                stop = trail_peak * (1 - trail)
                if lows[i] <= stop:
                    flag_ambiguous(ts)
                    sell(i, min(opens[i], stop), "trailing_stop_same_bar")
                    exited_this_bar = True
            elif highs[i] >= avg_price * (1 + tp_activation):
                trail_active = True
                trail_peak = highs[i]
                stop = trail_peak * (1 - trail)
                if lows[i] <= stop:
                    flag_ambiguous(ts)
                    sell(i, min(opens[i], stop), "trailing_stop_same_bar")
                    exited_this_bar = True
        if qty > 0:
            intrabar_notional = max(intrabar_notional, qty * mark_high[i])
            cycle_max_notional = max(cycle_max_notional, intrabar_notional)

        if halted:
            # Capture later prices for the equity trace while keeping new orders disabled.
            pass
        mark_price = mark_close[i]
        current_equity = balance + qty * (mark_price - avg_price) if qty > 0 else balance
        current_notional = qty * mark_price
        equity_rows.append((ts, current_equity, balance, qty, avg_price, current_notional, intrabar_notional,
                            p_values[i], current_on, halted, cycle_id))

    if qty > 0:
        final_eq = balance + qty * (mark_close[-1] - avg_price)
        finish_cycle(timestamps[-1], "open_at_test_end", final_eq)
    equity = pd.DataFrame(equity_rows, columns=["timestamp", "equity", "balance", "position_qty",
                            "average_price", "notional", "intrabar_max_notional", "p_unsafe", "ml_on", "halted", "cycle_id"]).set_index("timestamp")
    trades_frame = pd.DataFrame(trades)
    if not trades_frame.empty:
        trades_frame["ambiguous_bar"] = trades_frame["time"].isin(ambiguous_timestamps)
    cycles_frame = pd.DataFrame(cycles)
    if len(cycles_frame):
        cycles_frame.loc[cycles_frame["exit_reason"] == "open_at_test_end", "end_time"] = timestamps[-1]
    summary = summarize_backtest(equity, trades_frame, cycles_frame, start_equity)
    summary.update({
        "fees": fee_total, "slippage_cost": slip_total, "funding_cost": funding_total,
        "gross_realized_pnl": gross_total, "risk_exits": risk_exits,
        "modeled_liquidations": modeled_liquidations, "ambiguous_bars": ambiguous_bars,
        "missed_orders_injected": int((trades_frame.get("side", pd.Series(dtype=str)) == "SKIPPED").sum()),
        "execution_delay_bars": execution_delay_bars,
    })
    return BacktestResult(equity, trades_frame, cycles_frame, summary)


def summarize_backtest(equity: pd.DataFrame, trades: pd.DataFrame, cycles: pd.DataFrame,
                       initial_equity: float) -> dict:
    values = equity["equity"].astype(float)
    returns = values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    days = max((values.index[-1] - values.index[0]).total_seconds() / 86400, 1 / 24)
    years = days / 365.25
    peak = values.cummax()
    drawdown = values / peak - 1
    periods_per_year = 365.25 * 24 * 12
    mean = returns.mean() if len(returns) else np.nan
    std = returns.std(ddof=1) if len(returns) > 1 else np.nan
    downside = returns.clip(upper=0)
    downside_std = np.sqrt((downside ** 2).mean()) if len(downside) else np.nan
    final_equity = float(values.iloc[-1])
    total_return = final_equity / initial_equity - 1
    if final_equity > 0:
        try:
            cagr = (final_equity / initial_equity) ** (1 / years) - 1
        except OverflowError:
            cagr = math.inf
    else:
        cagr = -1.0
    if cycles.empty or "net_pnl" not in cycles.columns:
        profit_factor = np.nan
    else:
        net_cycles = cycles["net_pnl"].astype(float)
        gains = net_cycles[net_cycles > 0].sum()
        losses = -net_cycles[net_cycles < 0].sum()
        profit_factor = gains / losses if losses > 0 else (np.inf if gains > 0 else np.nan)
    cycle_losses = cycles["net_return"] if not cycles.empty else pd.Series(dtype=float)
    return {
        "start": values.index[0].isoformat(), "end": values.index[-1].isoformat(),
        "initial_equity": float(initial_equity), "final_equity": final_equity,
        "total_return": float(total_return), "cagr": float(cagr),
        "max_drawdown": float(drawdown.min()),
        "sharpe": float(mean / std * np.sqrt(periods_per_year)) if std and np.isfinite(std) else np.nan,
        "sortino": float(mean / downside_std * np.sqrt(periods_per_year)) if downside_std and np.isfinite(downside_std) else np.nan,
        "calmar": float(cagr / abs(drawdown.min())) if drawdown.min() < 0 else np.nan,
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else profit_factor,
        "cycles": int(len(cycles)),
        "worst_cycle_loss": float(cycle_losses.min()) if len(cycle_losses) else np.nan,
        "max_safety_depth": int(cycles["max_safety_depth"].max()) if len(cycles) else 0,
        "max_exposure_fraction_of_initial": float(equity["intrabar_max_notional"].max() / initial_equity),
        "max_exposure_fraction_of_cycle_start": (
            float((cycles["max_notional"].astype(float) / cycles["start_equity"].astype(float)).max())
            if len(cycles) and (cycles["start_equity"].astype(float) > 0).all() else np.nan
        ),
        "time_in_market": float((equity["position_qty"] > 0).mean()),
        "ml_off_fraction": float((~equity["ml_on"]).mean()) if equity["p_unsafe"].notna().any() else np.nan,
        "equity_floor_hits": int((trades.get("reason", pd.Series(dtype=str)) == "hard_equity_floor").sum()),
    }

