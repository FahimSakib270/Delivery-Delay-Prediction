"""CLI: ``python -m src.prepare`` - build the modelling table under data/processed/."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .config import load_config, resolve
from .data_prep import MODEL_FEATURES, prepare_dataset
from .features import feature_audit_table
from .utils import ensure_dir, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.prepare",
        description="Load, clean, join and label the delivery-delay dataset.",
    )
    parser.add_argument("--config", default=None, help="Alternative YAML config path.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    df, meta = prepare_dataset(cfg)

    reports_dir = ensure_dir(resolve(cfg, "paths", "reports_dir"))
    audit_path = reports_dir / "feature_audit.csv"
    feature_audit_table().to_csv(audit_path, index=False)
    save_json(meta, reports_dir / "data_prep_manifest.json")

    stats = meta["statistics"]
    print("")
    print("=" * 78)
    print("PREPARED DATASET")
    print("=" * 78)
    print(f"  source kind        : {stats['data_source_kind']}"
          f"{'  (SYNTHETIC DEMONSTRATION DATA)' if stats['synthetic'] else ''}")
    print(f"  orders (rows)      : {stats['final_orders']:,}   "
          f"({stats['item_rows_in']:,} item rows aggregated)")
    print(f"  booking window     : {stats['booking_period']['start'][:10]} .. "
          f"{stats['booking_period']['end'][:10]}")
    print(f"  late rate (primary): {stats['late_rate']:.4f}")
    print(f"  late rate (ts chk) : {stats['final_late_rate_timestamp_definition']:.4f}")
    print(f"  model features     : {len(MODEL_FEATURES)} "
          f"({len(stats['numeric_features'])} numeric, "
          f"{len(stats['categorical_features'])} categorical)")
    print(f"  written            : {stats['processed_path']}")
    print(f"  audit table        : {audit_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
