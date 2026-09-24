from __future__ import annotations

import argparse
import json

from .config import load_config
from .data import expected_months, ensure_supplemental_archives
from .research import prepare_data, run_research


def main() -> None:
    parser = argparse.ArgumentParser(prog="btc-dca", description="BTCUSDT ML-gated DCA research backtest")
    parser.add_argument("command", choices=["download", "validate", "run"])
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--skip-download", action="store_true", help="run with already downloaded mark/funding archives")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "download":
        counts = ensure_supplemental_archives(cfg.mark_dir, cfg.funding_dir,
                                               expected_months(cfg.start, cfg.end),
                                               verify=cfg.verify_official_checksums)
        print(json.dumps(counts, ensure_ascii=False, indent=2))
    elif args.command == "validate":
        _, _, _, _, quality = prepare_data(cfg, download_supplemental=not args.skip_download)
        print(json.dumps(quality, ensure_ascii=False, indent=2))
    elif args.command == "run":
        result = run_research(cfg, download_supplemental=not args.skip_download)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

