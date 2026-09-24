from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class AppConfig:
    source_path: Path
    ohlcv_dir: Path
    mark_dir: Path
    funding_dir: Path
    output_dir: Path
    start: str
    end: str
    verify_official_checksums: bool
    strategy: dict
    ml: dict


def load_config(path: str | Path = "config.toml") -> AppConfig:
    source = Path(path).resolve()
    raw = tomllib.loads(source.read_text(encoding="utf-8"))
    paths = raw["paths"]
    base = source.parent

    def resolve(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (base / p).resolve()

    return AppConfig(
        source_path=source,
        ohlcv_dir=resolve(paths["ohlcv_dir"]),
        mark_dir=resolve(paths["mark_dir"]),
        funding_dir=resolve(paths["funding_dir"]),
        output_dir=resolve(paths["output_dir"]),
        start=raw["data"]["start"],
        end=raw["data"]["end"],
        verify_official_checksums=bool(raw["data"].get("verify_official_checksums", True)),
        strategy=raw["strategy"],
        ml=raw["ml"],
    )


