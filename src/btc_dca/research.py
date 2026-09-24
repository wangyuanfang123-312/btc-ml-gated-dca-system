from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AppConfig
from .data import (expected_months, load_funding, load_klines, validate_klines,
                   verify_official_checksum, ensure_supplemental_archives,
                   patch_mark_gaps_from_daily)
from .features import aggregate_4h, build_features_and_label
from .engine import BacktestResult, run_backtest
from .walkforward import walk_forward, hysteresis
from .stress import run_stress_suite
from .report import build_report


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _verify_checksums(cfg: AppConfig, months: list[str]) -> dict[str, int]:
    paths = []
    for month in months:
        paths.extend([
            (cfg.ohlcv_dir / f"BTCUSDT-5m-{month}.zip", "ohlcv", month),
            (cfg.mark_dir / f"BTCUSDT-5m-{month}.zip", "mark", month),
            (cfg.funding_dir / f"BTCUSDT-fundingRate-{month}.zip", "funding", month),
        ])
    counts = {"ohlcv": 0, "mark": 0, "funding": 0}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(verify_official_checksum, path, family, month)
                   for path, family, month in paths]
        for (path, family, month), future in zip(paths, futures):
            future.result()
            counts[family] += 1
    return counts


def prepare_data(cfg: AppConfig, download_supplemental: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    months = expected_months(cfg.start, cfg.end)
    if download_supplemental:
        ensure_supplemental_archives(cfg.mark_dir, cfg.funding_dir, months,
                                     verify=cfg.verify_official_checksums)
    for family, folder, prefix in [
        ("OHLCV", cfg.ohlcv_dir, "BTCUSDT-5m-"),
        ("mark price", cfg.mark_dir, "BTCUSDT-5m-"),
        ("funding rate", cfg.funding_dir, "BTCUSDT-fundingRate-"),
    ]:
        missing = [m for m in months if not (folder / f"{prefix}{m}.zip").exists()]
        if missing:
            raise FileNotFoundError(f"Missing {family} monthly archives: {missing[:8]} (total {len(missing)}). Run `btc-dca download`.")

    ohlcv = load_klines(cfg.ohlcv_dir, months, "ohlcv")
    mark = load_klines(cfg.mark_dir, months, "mark", allow_incomplete_months=True)
    mark, daily_repair = patch_mark_gaps_from_daily(mark, ohlcv.index, cfg.mark_dir)
    mark = mark.reindex(ohlcv.index)
    funding = load_funding(cfg.funding_dir, months)
    checksum_count = _verify_checksums(cfg, months) if cfg.verify_official_checksums else {"ohlcv": 0, "mark": 0, "funding": 0}

    summaries = [
        validate_klines(ohlcv, "BTCUSDT USD-M execution kline 5m", len(months), checksum_count["ohlcv"]),
        validate_klines(mark, "BTCUSDT USD-M mark-price kline 5m", len(months), checksum_count["mark"]),
    ]
    if summaries[0].duplicate_timestamps or summaries[0].missing_5m_bars or summaries[0].invalid_ohlc_rows:
        raise ValueError(f"Quality gate failed for {summaries[0].source}: {summaries[0].as_dict()}")
    if not ohlcv.index.equals(mark.index):
        missing_mark = ohlcv.index.difference(mark.index)
        extra_mark = mark.index.difference(ohlcv.index)
        raise ValueError(f"Execution/mark timestamps differ: missing mark={len(missing_mark)}, extra mark={len(extra_mark)}")
    if funding.index.duplicated().any() or funding["funding_rate"].isna().any():
        raise ValueError("Funding-rate timestamps are duplicated or funding values are missing")
    if (funding["funding_rate"].abs() > 0.1).any():
        raise ValueError("Funding rate outside ±10%; verify archive schema and units")
    if not funding["interval_hours"].between(1, 24).all():
        raise ValueError("Funding interval outside expected 1–24 hour range")
    if not funding.index.isin(ohlcv.index).all():
        raise ValueError("Funding timestamps do not align to 5m execution bars")

    bars_4h = aggregate_4h(ohlcv)
    features = build_features_and_label(bars_4h,
        horizon_bars=int(cfg.ml["label_horizon_bars"]),
        unsafe_drawdown=float(cfg.ml["unsafe_drawdown"]))
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    ohlcv.to_parquet(cfg.output_dir / "ohlcv_5m.parquet", compression="zstd")
    mark.to_parquet(cfg.output_dir / "mark_5m.parquet", compression="zstd")
    funding.to_parquet(cfg.output_dir / "funding.parquet", compression="zstd")
    bars_4h.to_parquet(cfg.output_dir / "bars_4h.parquet", compression="zstd")
    features.to_parquet(cfg.output_dir / "features_4h.parquet", compression="zstd")
    quality = {
        "months": months,
        "ohlcv": summaries[0].as_dict(),
        "mark": summaries[1].as_dict(),
        "funding": {"rows": len(funding), "first_time": funding.index.min(),
                    "last_time": funding.index.max(), "interval_hours": sorted(funding["interval_hours"].unique().tolist()),
                    "max_timestamp_offset_ms": float(funding["timestamp_offset_ms"].max()),
                    "duplicate_timestamps": 0, "checksum_checked": checksum_count["funding"]},
        "bars_4h": len(bars_4h),
        "features_4h": len(features),
        "label_positive_rate": float(features["unsafe"].mean()),
        "mark_daily_repair": daily_repair,
        "oos_missing_mark_rows": int(mark.loc[(mark.index >= pd.Timestamp(cfg.ml["test_start"], tz="UTC")) &
                                                    (mark.index < pd.Timestamp(cfg.end, tz="UTC"))].isna().any(axis=1).sum()),
        "mark_observed_rows": int(mark["close"].notna().sum()),
        "mark_missing_timestamps": [t.isoformat() for t in mark.index[mark["close"].isna()]],
    }
    quality["mark_daily_repair"]["remaining_missing_rows"] = quality["mark_daily_repair"].get("remaining_missing_rows", 0) + int(mark["close"].isna().sum())
    quality["mark_daily_repair"]["remaining_missing_timestamps"] = quality["mark_missing_timestamps"]
    (cfg.output_dir / "data_quality.json").write_text(json.dumps(_json_safe(quality), indent=2), encoding="utf-8")
    return ohlcv, mark, funding, features, _json_safe(quality)


def _save_backtest(name: str, result: BacktestResult, output: Path) -> dict:
    result.equity.to_parquet(output / f"{name}_equity.parquet", compression="zstd")
    result.trades.to_csv(output / f"{name}_trades.csv", index=False, encoding="utf-8-sig")
    result.cycles.to_csv(output / f"{name}_cycles.csv", index=False, encoding="utf-8-sig")
    return {"strategy": name, **result.summary}


def _oos_period_comparison(folds: pd.DataFrame, results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Attribute continuous V1/V2 OOS curves to each walk-forward test window."""
    periods = (folds.loc[folds["model"] == "lightgbm", ["fold", "test_start", "test_end"]]
               .drop_duplicates().sort_values("test_start"))
    rows: list[dict] = []
    for _, period in periods.iterrows():
        start = pd.Timestamp(period["test_start"])
        end = pd.Timestamp(period["test_end"])
        row: dict = {"fold": int(period["fold"]), "test_start": start.isoformat(),
                     "test_end_exclusive": end.isoformat(),
                     "partial_period": end < start + pd.DateOffset(months=3)}
        for strategy in ("V1_oos", "V2_oos"):
            result = results[strategy]
            curve = result.equity
            inside = curve.loc[(curve.index >= start) & (curve.index < end), "equity"].astype(float)
            before = curve.loc[curve.index < start, "equity"]
            period_start_equity = float(before.iloc[-1]) if len(before) else float(result.summary["initial_equity"])
            period_end_equity = float(inside.iloc[-1]) if len(inside) else period_start_equity
            path = np.concatenate([[period_start_equity], inside.to_numpy(dtype=float)])
            local_drawdown = path / np.maximum.accumulate(path) - 1
            cycles = result.cycles.copy()
            cycle_ends = pd.to_datetime(cycles["end_time"], utc=True)
            completed = cycles.loc[(cycle_ends >= start) & (cycle_ends < end)
                                   & (cycles["exit_reason"] != "open_at_test_end")]
            row[f"{strategy}_return"] = period_end_equity / period_start_equity - 1 if period_start_equity else np.nan
            row[f"{strategy}_max_drawdown"] = float(local_drawdown.min()) if len(local_drawdown) else np.nan
            row[f"{strategy}_worst_completed_cycle"] = float(completed["net_return"].min()) if len(completed) else np.nan
            row[f"{strategy}_cycles_closed"] = int(len(completed))
            row[f"{strategy}_bars"] = int(len(inside))
        row["v2_minus_v1_return"] = row["V2_oos_return"] - row["V1_oos_return"]
        row["v2_minus_v1_max_drawdown"] = row["V2_oos_max_drawdown"] - row["V1_oos_max_drawdown"]
        row["v2_minus_v1_worst_completed_cycle"] = (
            row["V2_oos_worst_completed_cycle"] - row["V1_oos_worst_completed_cycle"]
            if np.isfinite(row["V1_oos_worst_completed_cycle"])
            and np.isfinite(row["V2_oos_worst_completed_cycle"]) else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_research(cfg: AppConfig, download_supplemental: bool = True) -> dict:
    ohlcv, mark, funding, features, quality = prepare_data(cfg, download_supplemental=download_supplemental)
    output = cfg.output_dir
    wf = walk_forward(
        features,
        test_start=cfg.ml["test_start"],
        train_months=int(cfg.ml["train_months"]),
        validation_months=int(cfg.ml["validation_months"]),
        test_months=int(cfg.ml["test_months"]),
        step_months=int(cfg.ml["step_months"]),
        purge_bars=int(cfg.ml["label_horizon_bars"]),
    )
    if wf.predictions.empty:
        raise RuntimeError("Walk-forward produced no out-of-sample predictions")
    wf.predictions.to_parquet(output / "oos_predictions.parquet", compression="zstd", index=False)
    wf.folds.to_csv(output / "walk_forward_metrics.csv", index=False, encoding="utf-8-sig")
    wf.feature_importance.to_csv(output / "lightgbm_feature_importance.csv", index=False, encoding="utf-8-sig")

    test_start = pd.Timestamp(cfg.ml["test_start"], tz="UTC")
    end = pd.Timestamp(cfg.end, tz="UTC")
    oos_ohlcv = ohlcv.loc[(ohlcv.index >= test_start) & (ohlcv.index < end)]
    oos_mark = mark.loc[oos_ohlcv.index]
    oos_funding = funding.loc[(funding.index >= test_start) & (funding.index < end)]
    lightgbm = wf.predictions.loc[wf.predictions["model"] == "lightgbm"].copy()
    signals = lightgbm.set_index("timestamp")[["p_unsafe"]].sort_index()
    if signals.index.duplicated().any():
        raise ValueError("Walk-forward generated duplicate test timestamps")
    signals["ml_on"] = hysteresis(signals["p_unsafe"], float(cfg.ml["on_threshold"]),
                                  float(cfg.ml["off_threshold"]), int(cfg.ml["on_confirmation_bars"]))

    summaries = []
    results: dict[str, BacktestResult] = {}
    # Full-history safe-DCA is useful context; all model comparisons use the common OOS interval.
    results["V1_full"] = run_backtest(ohlcv, cfg.strategy)
    summaries.append(_save_backtest("V1_full", results["V1_full"], output))
    results["V1_oos"] = run_backtest(oos_ohlcv, cfg.strategy)
    summaries.append(_save_backtest("V1_oos", results["V1_oos"], output))
    results["V2_oos"] = run_backtest(oos_ohlcv, cfg.strategy, signal=signals)
    summaries.append(_save_backtest("V2_oos", results["V2_oos"], output))
    mark_is_complete_for_oos = not oos_mark[["open", "high", "low", "close"]].isna().any().any()
    if mark_is_complete_for_oos:
        results["V1_realistic"] = run_backtest(oos_ohlcv, cfg.strategy, mark=oos_mark,
                                               funding=oos_funding, use_mark_for_risk=True)
        summaries.append(_save_backtest("V1_realistic", results["V1_realistic"], output))
        results["V3_realistic"] = run_backtest(oos_ohlcv, cfg.strategy, signal=signals,
                                               mark=oos_mark, funding=oos_funding,
                                               use_mark_for_risk=True)
        summaries.append(_save_backtest("V3_realistic", results["V3_realistic"], output))
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output / "strategy_comparison.csv", index=False, encoding="utf-8-sig")
    (output / "strategy_comparison.json").write_text(json.dumps(_json_safe(summaries), indent=2), encoding="utf-8")

    period_comparison = _oos_period_comparison(wf.folds, results)
    period_comparison.to_csv(output / "oos_period_comparison.csv", index=False, encoding="utf-8-sig")

    stress_base = results.get("V3_realistic", results["V2_oos"])
    stress = run_stress_suite(oos_ohlcv, oos_mark, oos_funding, signals, cfg.strategy,
                              stress_base, use_mark_for_risk=mark_is_complete_for_oos)
    stress.to_csv(output / "stress_results.csv", index=False, encoding="utf-8-sig")
    build_report(output, quality, summary_frame, wf.folds, wf.feature_importance, stress,
                 predictions=wf.predictions, periods=period_comparison)
    return {"quality": quality, "summary": summary_frame, "walk_forward_folds": len(wf.folds),
            "stress_scenarios": len(stress), "v3_status": "ran" if mark_is_complete_for_oos else "blocked: missing official mark bars in OOS",
            "output_dir": str(output.resolve())}

