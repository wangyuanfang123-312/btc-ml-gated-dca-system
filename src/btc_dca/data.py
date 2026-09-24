from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd


BASE_URL = "https://data.binance.vision/data/futures/um/monthly"
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
MARK_COLUMNS = KLINE_COLUMNS


@dataclass
class QualitySummary:
    source: str
    rows: int
    files: int
    first_time: str
    last_time: str
    duplicate_timestamps: int
    missing_5m_bars: int
    invalid_ohlc_rows: int
    zero_volume_rows: int
    checksum_checked: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def expected_months(start: str, end_exclusive: str) -> list[str]:
    periods = pd.period_range(pd.Timestamp(start), pd.Timestamp(end_exclusive) - pd.Timedelta(days=1), freq="M")
    return [p.strftime("%Y-%m") for p in periods]


def _month_zip_path(folder: Path, family: str, month: str) -> Path:
    if family == "ohlcv":
        return folder / f"BTCUSDT-5m-{month}.zip"
    if family == "mark":
        return folder / f"BTCUSDT-5m-{month}.zip"
    if family == "funding":
        return folder / f"BTCUSDT-fundingRate-{month}.zip"
    raise ValueError(f"Unsupported data family: {family}")


def _remote_path(family: str, month: str) -> str:
    if family == "ohlcv":
        return f"klines/BTCUSDT/5m/BTCUSDT-5m-{month}.zip"
    if family == "mark":
        return f"markPriceKlines/BTCUSDT/5m/BTCUSDT-5m-{month}.zip"
    if family == "funding":
        return f"fundingRate/BTCUSDT/BTCUSDT-fundingRate-{month}.zip"
    raise ValueError(f"Unsupported data family: {family}")


def _read_zip_csv(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Expected one CSV in {path}, found {names}")
        with archive.open(names[0]) as stream:
            frame = pd.read_csv(stream, header=None)
    return frame


def load_klines(folder: Path, months: list[str], family: str = "ohlcv",
                allow_incomplete_months: bool = False) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for month in months:
        path = _month_zip_path(folder, family, month)
        if not path.exists():
            raise FileNotFoundError(f"Missing {family} archive: {path}")
        frame = _read_zip_csv(path)
        if len(frame.columns) != 12:
            raise ValueError(f"Unexpected 12-column kline schema in {path}: {len(frame.columns)} columns")
        frame.columns = KLINE_COLUMNS
        # Binance monthly archives are not consistent about whether they include a header.
        frame["open_time"] = pd.to_numeric(frame["open_time"], errors="coerce")
        frame = frame.dropna(subset=["open_time"]).copy()
        frame["open_time"] = frame["open_time"].astype("int64")
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        month_start = pd.Timestamp(f"{month}-01", tz="UTC")
        month_end = month_start + pd.offsets.MonthBegin(1)
        expected_rows = int((month_end - month_start).total_seconds() // 300)
        expected_first = month_start.value // 1_000_000
        expected_last = expected_first + (expected_rows - 1) * 300_000
        actual_first = int(frame["open_time"].min()) if len(frame) else None
        actual_last = int(frame["open_time"].max()) if len(frame) else None
        if not allow_incomplete_months and (len(frame) != expected_rows or actual_first != expected_first or actual_last != expected_last):
            raise ValueError(f"Incomplete monthly {family} kline archive {path.name}: rows={len(frame)} expected={expected_rows}, first={actual_first}, last={actual_last}")
        frames.append(frame[["open_time", "open", "high", "low", "close", "volume"]])
    result = pd.concat(frames, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result.pop("open_time"), unit="ms", utc=True)
    result = result.sort_values("timestamp", kind="stable").set_index("timestamp")
    return result


def patch_mark_gaps_from_daily(mark: pd.DataFrame, expected_index: pd.DatetimeIndex,
                               folder: Path) -> tuple[pd.DataFrame, dict]:
    """Fill only absent monthly mark bars with checksum-verified official daily records."""
    missing = expected_index.difference(mark.index)
    if missing.empty:
        return mark, {"daily_days_checked": 0, "daily_rows_added": 0, "unavailable_days": []}
    daily_dir = folder / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    days = sorted({stamp.strftime("%Y-%m-%d") for stamp in missing})
    additions: list[pd.DataFrame] = []
    unavailable: list[str] = []
    added = 0
    for day in days:
        name = f"BTCUSDT-5m-{day}.zip"
        local = daily_dir / name
        relative = f"markPriceKlines/BTCUSDT/5m/{name}"
        url = f"https://data.binance.vision/data/futures/um/daily/{relative}"
        if not local.exists():
            try:
                with urllib.request.urlopen(url, timeout=45) as response:
                    payload = response.read()
                temp = local.with_suffix(local.suffix + ".part")
                temp.write_bytes(payload)
                with zipfile.ZipFile(temp) as archive:
                    if len([n for n in archive.namelist() if n.lower().endswith(".csv")]) != 1:
                        raise ValueError(f"Unexpected daily mark archive layout: {name}")
                temp.replace(local)
            except Exception as exc:
                unavailable.append(f"{day}: {type(exc).__name__}: {exc}")
                continue
        checksum_url = f"{url}.CHECKSUM"
        with urllib.request.urlopen(checksum_url, timeout=30) as response:
            expected_hash = response.read().decode("utf-8").split()[0].lower()
        if sha256(local.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Binance daily mark checksum mismatch for {name}")
        daily = _read_zip_csv(local)
        if len(daily.columns) != 12:
            raise ValueError(f"Unexpected daily mark kline schema in {name}")
        daily.columns = KLINE_COLUMNS
        daily["open_time"] = pd.to_numeric(daily["open_time"], errors="coerce")
        daily = daily.dropna(subset=["open_time"]).copy()
        daily["open_time"] = daily["open_time"].astype("int64")
        for column in ["open", "high", "low", "close", "volume"]:
            daily[column] = pd.to_numeric(daily[column], errors="coerce")
        daily["timestamp"] = pd.to_datetime(daily.pop("open_time"), unit="ms", utc=True)
        daily = daily.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        fill = daily.loc[daily.index.intersection(missing)]
        if not fill.empty:
            additions.append(fill)
            added += len(fill)
    if additions:
        patched = pd.concat([mark, *additions]).sort_index()
        if patched.index.duplicated().any():
            raise ValueError("Daily mark patches overlap existing monthly timestamps")
    else:
        patched = mark
    return patched, {"daily_days_checked": len(days) - len(unavailable), "daily_rows_added": added,
                     "unavailable_days": unavailable, "daily_days_requested": days}


def load_funding(folder: Path, months: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for month in months:
        path = _month_zip_path(folder, "funding", month)
        if not path.exists():
            raise FileNotFoundError(f"Missing funding archive: {path}")
        frame = _read_zip_csv(path)
        # Official monthly funding archives use calc_time, funding_interval_hours, last_funding_rate.
        if len(frame.columns) == 3:
            frame.columns = ["timestamp_ms", "interval_hours", "funding_rate"]
        elif len(frame.columns) >= 3:
            frame = frame.iloc[:, :3]
            frame.columns = ["timestamp_ms", "interval_hours", "funding_rate"]
        frame["timestamp_ms"] = pd.to_numeric(frame["timestamp_ms"], errors="coerce")
        frame["interval_hours"] = pd.to_numeric(frame["interval_hours"], errors="coerce")
        frame["funding_rate"] = pd.to_numeric(frame["funding_rate"], errors="coerce")
        frame = frame.dropna(subset=["timestamp_ms", "interval_hours", "funding_rate"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp_ms"].astype("int64"), unit="ms", utc=True)
        frame["timestamp_offset_ms"] = (frame["timestamp"] - frame["timestamp"].dt.floor("5min")).dt.total_seconds() * 1000
        frame["timestamp"] = frame["timestamp"].dt.floor("5min")
        frames.append(frame[["timestamp", "funding_rate", "interval_hours", "timestamp_offset_ms"]])
    result = pd.concat(frames, ignore_index=True).sort_values("timestamp", kind="stable")
    if result["timestamp"].duplicated().any():
        raise ValueError("Funding monthly archives contain duplicate funding timestamps")
    return result.set_index("timestamp")


def validate_klines(frame: pd.DataFrame, source: str, file_count: int, checksum_checked: int = 0) -> QualitySummary:
    if frame.empty:
        raise ValueError(f"{source}: no observations")
    times = frame.index
    duplicates = int(times.duplicated().sum())
    delta = times.to_series().diff().dropna()
    missing = int(((delta / pd.Timedelta(minutes=5)).round().astype("int64") - 1).clip(lower=0).sum())
    o, h, low, c = (frame[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    valid = (np.isfinite(o) & np.isfinite(h) & np.isfinite(low) & np.isfinite(c)
             & (np.minimum.reduce([o, h, low, c]) > 0)
             & (h >= np.maximum.reduce([o, c, low]))
             & (low <= np.minimum.reduce([o, c, h]))
             & (frame["volume"].to_numpy(dtype=float) >= 0))
    invalid = int((~valid).sum())
    return QualitySummary(
        source=source,
        rows=len(frame),
        files=file_count,
        first_time=times[0].isoformat(),
        last_time=times[-1].isoformat(),
        duplicate_timestamps=duplicates,
        missing_5m_bars=missing,
        invalid_ohlc_rows=invalid,
        zero_volume_rows=int((frame["volume"] == 0).sum()),
        checksum_checked=checksum_checked,
    )


def verify_official_checksum(path: Path, family: str, month: str) -> None:
    remote = _remote_path(family, month)
    checksum_url = f"{BASE_URL}/{remote}.CHECKSUM"
    with urllib.request.urlopen(checksum_url, timeout=30) as response:
        expected = response.read().decode("utf-8").split()[0].lower()
    actual = sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Binance checksum mismatch for {path.name}: expected {expected}, got {actual}")


def ensure_supplemental_archives(mark_dir: Path, funding_dir: Path, months: list[str], verify: bool = True) -> dict[str, int]:
    tasks = [(mark_dir, "mark", m) for m in months] + [(funding_dir, "funding", m) for m in months]
    mark_dir.mkdir(parents=True, exist_ok=True)
    funding_dir.mkdir(parents=True, exist_ok=True)

    def one(task: tuple[Path, str, str]) -> tuple[str, bool]:
        folder, family, month = task
        target = _month_zip_path(folder, family, month)
        if not target.exists():
            url = f"{BASE_URL}/{_remote_path(family, month)}"
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            if len(payload) < 100:
                raise ValueError(f"Downloaded archive is unexpectedly small: {url}")
            temp = target.with_suffix(target.suffix + ".part")
            temp.write_bytes(payload)
            try:
                with zipfile.ZipFile(temp) as archive:
                    if not archive.namelist():
                        raise ValueError(f"Empty archive: {url}")
                temp.replace(target)
            except Exception:
                if temp.exists():
                    temp.unlink()
                raise
        if verify:
            verify_official_checksum(target, family, month)
        return family, target.exists()

    done = {"mark": 0, "funding": 0}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(one, task) for task in tasks]
        for future in as_completed(futures):
            family, _ = future.result()
            done[family] += 1
    return done
