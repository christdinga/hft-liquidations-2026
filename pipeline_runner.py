"""
pipeline_runner.py
==================
Integrates the DatasetBuilder with the existing liquidation filter pipeline
(CHRIST_DINGA_TASK_2.py). Provides a single entry point to:

  1. Load data (reusing load_data_with_required_preprocess)
  2. Build a supervised dataset via the modular feature pipeline
  3. Train a model on the dataset
  4. Evaluate on train and validation splits

This file is standalone — it imports from feature_pipeline.py and
(optionally) from the original task file for data loading.

Usage:
    python pipeline_runner.py [--ml] [--sampler=volume|time|trade]
                              [--tau=30] [--limit=500000]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Import from the modular feature pipeline
from feature_pipeline import (
    DEFAULT_TRANSFORMS,
    BidAskSpreadFeature,
    DatasetBuilder,
    LiquidationPressureFeature,
    MarkoutLabeler,
    ModelSpec,
    PipelineConfig,
    PriceZScoreFeature,
    RollingVolatilityFeature,
    TimeClock,
    TimeFeature,
    TopOfBookVolumeFeature,
    TradeClock,
    TradePressureFeature,
    Transform,
    TransformLayer,
    VolumeClock,
    evaluate_model,
    train_model,
)

logger = logging.getLogger(__name__)

BYBIT_LAG_US: int = 200_000
SYMBOLS = ("btcusdt", "ethusdt")
SPLIT_RANGES = {
    "train": (
        pd.Timestamp("2025-12-01", tz="UTC"),
        pd.Timestamp("2026-02-01", tz="UTC"),
    ),
    "validation": (
        pd.Timestamp("2026-02-01", tz="UTC"),
        pd.Timestamp("2026-03-01", tz="UTC"),
    ),
}


# ---------------------------------------------------------------------------
# Data loading (thin wrapper around the original loader)
# ---------------------------------------------------------------------------


def load_data(
    data_dir: str,
    symbol: str,
    split: str | None = None,
    limit: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the 4 frames and apply mandatory preprocessing (Bybit shift).
    Tries to import from the original task file; falls back to a standalone loader.
    """
    try:
        # Reuse the battle-tested loader from the original pipeline
        from CHRIST_DINGA_TASK_2 import load_data_with_required_preprocess

        return load_data_with_required_preprocess(
            data_dir, symbol, split=split, limit=limit
        )
    except ImportError:
        logger.warning("Original task file not found — using standalone loader")
        return _standalone_loader(data_dir, symbol, split=split, limit=limit)


def _standalone_loader(
    data_dir: str,
    symbol: str,
    split: str | None = None,
    limit: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Minimal parquet loader (used if CHRIST_DINGA_TASK_2.py not present)."""
    import os

    import duckdb

    def _ts_to_us(ts: pd.Timestamp) -> int:
        return ts.value // 1000

    def _load(path: str, where: str = "") -> pd.DataFrame:
        q = f"SELECT * FROM read_parquet('{path}')"
        if where:
            q += f" WHERE {where}"
        if limit > 0:
            q += f" LIMIT {limit}"
        df = duckdb.sql(q).df()
        if "timestamp" in df.columns:
            df["timestamp"] = df["timestamp"].astype(np.int64)
        return df

    where = ""
    if split and split in SPLIT_RANGES:
        s, e = SPLIT_RANGES[split]
        where = f"timestamp >= {_ts_to_us(s)} AND timestamp < {_ts_to_us(e)}"

    sym = symbol.lower()
    base = Path(data_dir)

    trades = _load(str(base / "binance_trades" / f"perp_{sym}.parquet"), where)
    bbo = _load(str(base / "binance_booktickers" / f"perp_{sym}.parquet"), where)
    liq_b = _load(str(base / "binance_liquidations" / f"perp_{sym}.parquet"), where)
    liq_bb = _load(str(base / "bybit_liquidations" / f"{sym}.parquet"), where)

    # Normalize tickers
    for df in (trades, bbo, liq_b, liq_bb):
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].str.replace("perp:", "", regex=False)

    # Shift Bybit
    if len(liq_bb) > 0:
        liq_bb = liq_bb.copy()
        liq_bb["timestamp"] = liq_bb["timestamp"].astype(np.int64) + BYBIT_LAG_US

    for df in (trades, bbo, liq_b, liq_bb):
        if len(df) > 0:
            df.sort_values("timestamp", inplace=True, ignore_index=True)

    return trades, bbo, liq_b, liq_bb


# ---------------------------------------------------------------------------
# Pipeline configuration factory
# ---------------------------------------------------------------------------


def make_pipeline_config(
    sampler_type: str = "volume",
    volume_threshold: float = 100_000.0,
    time_interval_s: float = 60.0,
    trade_nth: int = 1,
    taus: list[float] = None,
    apply_transforms: bool = True,
) -> PipelineConfig:
    """
    Factory for PipelineConfig. Supports volume-clock, time-clock, trade-clock.
    Declarative: change sampler_type to switch sampling strategy.
    """
    taus = taus or [30.0, 120.0, 300.0]

    if sampler_type == "volume":
        sampler = VolumeClock(threshold=volume_threshold)
    elif sampler_type == "time":
        sampler = TimeClock(interval_s=time_interval_s)
    elif sampler_type == "trade":
        sampler = TradeClock(nth=trade_nth)
    else:
        raise ValueError(
            f"Unknown sampler_type: {sampler_type!r}. Choose volume/time/trade"
        )

    return PipelineConfig(
        features=[
            TopOfBookVolumeFeature(),
            PriceZScoreFeature(windows_s=(10.0, 60.0, 300.0), clip=5.0),
            RollingVolatilityFeature(windows_s=(30.0, 300.0)),
            BidAskSpreadFeature(),
            TradePressureFeature(windows_s=(5.0, 30.0, 120.0)),
            LiquidationPressureFeature(windows_s=(5.0, 30.0)),
            TimeFeature(),
        ],
        transforms=DEFAULT_TRANSFORMS,
        sampler=sampler,
        labeler=MarkoutLabeler(tau_s=taus),
        apply_transforms=apply_transforms,
    )


# ---------------------------------------------------------------------------
# Full end-to-end run
# ---------------------------------------------------------------------------


def run_full_pipeline(
    data_dir: str,
    sampler_type: str = "volume",
    taus: list[float] = None,
    use_ml: bool = True,
    limit: int = 100_000,
    primary_tau: float = 30.0,
) -> dict[str, Any]:
    """
    End-to-end pipeline:
      1. Load data for all symbols
      2. Build dataset (train + validation)
      3. Train one model per symbol × tau
      4. Evaluate and return reports

    Returns:
        {
          "train_datasets": {symbol: df},
          "val_datasets":   {symbol: df},
          "models":         {(symbol, tau): model},
          "reports":        {(symbol, tau): metrics_dict},
          "feature_names":  list[str],
        }
    """
    taus = taus or [30.0, 120.0, 300.0]
    config = make_pipeline_config(sampler_type=sampler_type, taus=taus)
    builder = DatasetBuilder(config)
    feature_names = builder.feature_names()

    train_datasets: dict[str, pd.DataFrame] = {}
    val_datasets: dict[str, pd.DataFrame] = {}
    models: dict[tuple[str, float], Any] = {}
    reports: dict[tuple[str, float], dict] = {}

    for symbol in SYMBOLS:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing symbol: {symbol.upper()}")
        logger.info(f"{'=' * 60}")

        # --- Load train ---
        logger.info(f"Loading train data...")
        trades_tr, bbo_tr, liq_b_tr, liq_bb_tr = load_data(
            data_dir, symbol, split="train", limit=limit
        )
        if len(trades_tr) == 0:
            logger.warning(f"No train data for {symbol}")
            continue

        ds_train = builder.build(trades_tr, bbo_tr, liq_b_tr, liq_bb_tr, symbol=symbol)
        train_datasets[symbol] = ds_train
        logger.info(
            f"Train dataset: {len(ds_train):,} rows × {len(feature_names)} features"
        )

        # --- Load validation ---
        logger.info(f"Loading validation data...")
        trades_val, bbo_val, liq_b_val, liq_bb_val = load_data(
            data_dir, symbol, split="validation", limit=limit
        )
        if len(trades_val) > 0:
            ds_val = builder.build(
                trades_val, bbo_val, liq_b_val, liq_bb_val, symbol=symbol
            )
            val_datasets[symbol] = ds_val
            logger.info(
                f"Val dataset: {len(ds_val):,} rows × {len(feature_names)} features"
            )
        else:
            logger.warning(f"No validation data for {symbol}")
            ds_val = pd.DataFrame()

        if len(ds_train) == 0:
            logger.warning(f"Empty train dataset for {symbol}, skipping")
            continue

        # --- Train one model per tau ---
        for tau in taus:
            target_col = f"pnl_bps_{int(tau)}s"
            if target_col not in ds_train.columns:
                logger.warning(f"Target {target_col} not in dataset, skipping")
                continue

            spec = ModelSpec(
                model_type="lgbm" if use_ml else "linear",
                target_col=target_col,
                use_sample_weights=True,
            )

            try:
                model = train_model(ds_train, feature_names, spec)
                models[(symbol, tau)] = model

                # Evaluate on train
                tr_metrics = evaluate_model(model, ds_train, feature_names, spec)
                tr_metrics["split"] = "train"

                # Evaluate on validation
                if len(ds_val) > 0 and target_col in ds_val.columns:
                    val_metrics = evaluate_model(model, ds_val, feature_names, spec)
                    val_metrics["split"] = "validation"
                else:
                    val_metrics = {}

                reports[(symbol, tau)] = {"train": tr_metrics, "val": val_metrics}

                logger.info(
                    f"  [{symbol} τ={int(tau)}s] "
                    f"train IC={tr_metrics.get('ic_spearman', 0):.4f} "
                    f"wcorr={tr_metrics.get('weighted_corr', 0):.4f} | "
                    f"val IC={val_metrics.get('ic_spearman', 0):.4f} "
                    f"wcorr={val_metrics.get('weighted_corr', 0):.4f}"
                )
            except Exception as e:
                logger.error(f"  [{symbol} τ={int(tau)}s] Training failed: {e}")

    return {
        "train_datasets": train_datasets,
        "val_datasets": val_datasets,
        "models": models,
        "reports": reports,
        "feature_names": feature_names,
        "config": config,
    }


def print_report(results: dict[str, Any]) -> None:
    """Pretty-print training results."""
    reports = results["reports"]
    if not reports:
        print("No results to display.")
        return

    print(f"\n{'─' * 80}")
    print(
        f"  {'Symbol':<10} {'τ':>5}  {'Train IC':>10} {'Train Corr':>12} {'Val IC':>10} {'Val Corr':>10} {'N_train':>8}"
    )
    print(f"{'─' * 80}")

    for (symbol, tau), r in sorted(reports.items()):
        tr = r.get("train", {})
        vl = r.get("val", {})
        print(
            f"  {symbol.upper():<10} {int(tau):>4}s"
            f"  {tr.get('ic_spearman', float('nan')):>+10.4f}"
            f"  {tr.get('weighted_corr', float('nan')):>+12.4f}"
            f"  {vl.get('ic_spearman', float('nan')):>+10.4f}"
            f"  {vl.get('weighted_corr', float('nan')):>+10.4f}"
            f"  {tr.get('n_samples', 0):>8,}"
        )
    print(f"{'─' * 80}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir = "data"
    sampler_type = "volume"
    use_ml = True
    limit = 100_000
    taus = [30.0, 120.0, 300.0]

    for arg in sys.argv[1:]:
        if arg == "--ml":
            use_ml = True
        elif arg == "--no-ml":
            use_ml = False
        elif arg.startswith("--sampler="):
            sampler_type = arg.split("=", 1)[1]
        elif arg.startswith("--limit="):
            limit = int(arg.split("=", 1)[1])
        elif arg.startswith("--tau="):
            taus = [float(x) for x in arg.split("=", 1)[1].split(",")]

    print(f"\n{'=' * 80}")
    print(f"  Modular Feature Pipeline — Supervised Dataset Builder")
    print(f"  Sampler:  {sampler_type}")
    print(f"  Model:    {'LightGBM' if use_ml else 'Ridge'}")
    print(f"  Taus:     {taus}")
    print(f"  Limit:    {limit:,} trades/symbol")
    print(f"{'=' * 80}\n")

    results = run_full_pipeline(
        data_dir=data_dir,
        sampler_type=sampler_type,
        taus=taus,
        use_ml=use_ml,
        limit=limit,
    )

    print_report(results)
    print("[OK] Pipeline complete.")
