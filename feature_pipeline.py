"""
feature_pipeline.py
===================
Modular, composable feature pipeline for market microstructure ML.

Architecture:
    raw market data
        → FeatureBlock (per-trade feature computation)
        → TransformLayer (log / zscore / clamp applied per feature subset)
        → FeatureVector assembly
        → Sampler (volume-clock or time-clock or trade-clock)
        → Labeler (n-second markout of the EXECUTED trade)
        → DatasetOutput
        → ModelTrainer

Design principles:
  - Each block is independently testable and reusable
  - Features declared as a list, not hardcoded
  - Transforms declared as reusable operators
  - Dataset construction is deterministic and reproducible
  - Model training is decoupled from feature computation
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

US_PER_SECOND: int = 1_000_000
NOTIONAL_CLIP: float = 100_000.0

# ---------------------------------------------------------------------------
# 1. FEATURE BLOCK ABSTRACTIONS
# ---------------------------------------------------------------------------


class FeatureBlock(ABC):
    """
    Base class for a single feature generator.

    Each block:
      - declares its output column names via `output_columns`
      - is called once per sampled trade via `compute(state, trade_row)`,
        where `state` holds references to sorted arrays of past data
    """

    @property
    @abstractmethod
    def output_columns(self) -> list[str]:
        """Names of columns this block produces."""
        ...

    @abstractmethod
    def compute(self, state: "PipelineState", trade_idx: int) -> dict[str, float]:
        """
        Compute features for trade at index `trade_idx` in state.trades.
        Must use ONLY data with timestamp < trades.timestamp[trade_idx].
        Returns {column_name: value}.
        """
        ...


@dataclass
class PipelineState:
    """
    Shared market state passed to all FeatureBlocks.
    Pre-sorted arrays enable O(log n) causal lookups.
    """

    # Trades (full frame, sorted by timestamp)
    trades: pd.DataFrame

    # BBO sorted arrays
    bbo_ts: np.ndarray  # int64 us
    bbo_bid: np.ndarray  # float64
    bbo_ask: np.ndarray  # float64
    bbo_bid_amt: np.ndarray  # float64
    bbo_ask_amt: np.ndarray  # float64
    bbo_mid: np.ndarray  # float64

    # Liquidation sorted arrays (already shifted)
    liq_ts: np.ndarray  # int64 us (combined Binance + Bybit)
    liq_side: np.ndarray  # str array ("buy" / "sell")
    liq_notional: np.ndarray  # float64

    # Metadata
    symbol: str = "unknown"

    @classmethod
    def build(
        cls,
        trades: pd.DataFrame,
        bbo: pd.DataFrame,
        liq_binance: pd.DataFrame,
        liq_bybit: pd.DataFrame,
        symbol: str = "unknown",
    ) -> "PipelineState":
        """Construct from raw frames (assumes Bybit already shifted)."""
        bbo_s = bbo.sort_values("timestamp").reset_index(drop=True)
        mid = (bbo_s["bid_price"] + bbo_s["ask_price"]) / 2.0

        # Combine liquidations
        frames = []
        for df in (liq_binance, liq_bybit):
            if df is not None and len(df) > 0:
                frames.append(df[["timestamp", "side", "price", "amount"]].copy())
        if frames:
            liq_all = pd.concat(frames, ignore_index=True).sort_values("timestamp")
            liq_ts = liq_all["timestamp"].values.astype(np.int64)
            liq_side = liq_all["side"].str.lower().values
            liq_notional = (liq_all["price"] * liq_all["amount"]).values.astype(
                np.float64
            )
        else:
            liq_ts = np.array([], dtype=np.int64)
            liq_side = np.array([], dtype=object)
            liq_notional = np.array([], dtype=np.float64)

        return cls(
            trades=trades.sort_values("timestamp").reset_index(drop=True),
            bbo_ts=bbo_s["timestamp"].values.astype(np.int64),
            bbo_bid=bbo_s["bid_price"].values.astype(np.float64),
            bbo_ask=bbo_s["ask_price"].values.astype(np.float64),
            bbo_bid_amt=bbo_s["bid_amount"].values.astype(np.float64),
            bbo_ask_amt=bbo_s["ask_amount"].values.astype(np.float64),
            bbo_mid=mid.values.astype(np.float64),
            liq_ts=liq_ts,
            liq_side=liq_side,
            liq_notional=liq_notional,
            symbol=symbol,
        )

    # --- lookup helpers (strictly causal: ts < query_ts) ---

    def last_bbo_idx(self, query_ts: int) -> int:
        """Index of last BBO row with timestamp < query_ts. -1 if none."""
        idx = int(np.searchsorted(self.bbo_ts, query_ts, side="left")) - 1
        return idx

    def bbo_window_slice(self, query_ts: int, window_s: float) -> slice:
        """Slice of BBO rows in (query_ts - window_us, query_ts)."""
        win = int(window_s * US_PER_SECOND)
        lo = int(np.searchsorted(self.bbo_ts, query_ts - win, side="right"))
        hi = int(np.searchsorted(self.bbo_ts, query_ts, side="left"))
        return slice(lo, hi)

    def liq_window_slice(
        self, query_ts: int, window_s: float, side: str | None = None
    ) -> np.ndarray:
        """Boolean mask of liq events in (query_ts - window_us, query_ts)."""
        win = int(window_s * US_PER_SECOND)
        lo = int(np.searchsorted(self.liq_ts, query_ts - win, side="right"))
        hi = int(np.searchsorted(self.liq_ts, query_ts, side="left"))
        if lo >= hi:
            return np.array([], dtype=np.float64), np.array([], dtype=object)
        ns = self.liq_notional[lo:hi]
        ss = self.liq_side[lo:hi]
        if side is not None:
            mask = ss == side
            return ns[mask], ss[mask]
        return ns, ss

    def trade_window_slice(self, query_ts: int, window_s: float):
        """Trades in (query_ts - window_us, query_ts)."""
        win = int(window_s * US_PER_SECOND)
        ts_arr = self.trades["timestamp"].values.astype(np.int64)
        lo = int(np.searchsorted(ts_arr, query_ts - win, side="right"))
        hi = int(np.searchsorted(ts_arr, query_ts, side="left"))
        return self.trades.iloc[lo:hi]


# ---------------------------------------------------------------------------
# 2. CONCRETE FEATURE BLOCKS
# ---------------------------------------------------------------------------


class TopOfBookVolumeFeature(FeatureBlock):
    """
    Top-of-book notional volume: bid_amount * bid_price, ask_amount * ask_price.
    Also computes book imbalance (inherently scale-free).
    """

    @property
    def output_columns(self) -> list[str]:
        return ["tob_bid_notional", "tob_ask_notional", "tob_imbalance"]

    def compute(self, state: PipelineState, trade_idx: int) -> dict[str, float]:
        qt = int(state.trades["timestamp"].iloc[trade_idx])
        idx = state.last_bbo_idx(qt)
        if idx < 0:
            return {c: np.nan for c in self.output_columns}

        mid = state.bbo_mid[idx]
        bid_n = state.bbo_bid_amt[idx] * state.bbo_bid[idx]
        ask_n = state.bbo_ask_amt[idx] * state.bbo_ask[idx]
        total = bid_n + ask_n
        imbalance = (bid_n - ask_n) / total if total > 0 else 0.0

        return {
            "tob_bid_notional": bid_n,
            "tob_ask_notional": ask_n,
            "tob_imbalance": imbalance,
        }


class PriceZScoreFeature(FeatureBlock):
    """
    Price z-score over multiple horizons: (price - mean_t) / std_t.
    Output is inherently scale-free. Clipped to [-5, 5].
    """

    def __init__(
        self, windows_s: tuple[float, ...] = (10.0, 60.0, 300.0), clip: float = 5.0
    ):
        self.windows_s = windows_s
        self.clip = clip

    @property
    def output_columns(self) -> list[str]:
        return [f"price_zscore_{w}s" for w in self.windows_s]

    def compute(self, state: PipelineState, trade_idx: int) -> dict[str, float]:
        qt = int(state.trades["timestamp"].iloc[trade_idx])
        price = float(state.trades["price"].iloc[trade_idx])
        out = {}

        for w in self.windows_s:
            sl = state.bbo_window_slice(qt, w)
            mids = state.bbo_mid[sl]
            col = f"price_zscore_{w}s"
            if len(mids) < 2:
                out[col] = 0.0
            else:
                mu = mids.mean()
                sigma = mids.std()
                z = (price - mu) / sigma if sigma > 1e-10 else 0.0
                out[col] = float(np.clip(z, -self.clip, self.clip))

        return out


class RollingVolatilityFeature(FeatureBlock):
    """
    Rolling price volatility = std(mid_returns) / mid — normalized by price level.
    """

    def __init__(self, windows_s: tuple[float, ...] = (30.0, 300.0)):
        self.windows_s = windows_s

    @property
    def output_columns(self) -> list[str]:
        return [f"realized_vol_{w}s" for w in self.windows_s]

    def compute(self, state: PipelineState, trade_idx: int) -> dict[str, float]:
        qt = int(state.trades["timestamp"].iloc[trade_idx])
        out = {}
        for w in self.windows_s:
            sl = state.bbo_window_slice(qt, w)
            mids = state.bbo_mid[sl]
            col = f"realized_vol_{w}s"
            if len(mids) < 2:
                out[col] = 0.0
            else:
                returns = np.diff(mids) / mids[:-1]
                # Normalize by mid level → scale-free
                out[col] = float(returns.std())
        return out


class BidAskSpreadFeature(FeatureBlock):
    """
    Bid-ask spread normalized by mid-price: (ask - bid) / mid * 10000 (bps).
    Scale-free by construction.
    """

    @property
    def output_columns(self) -> list[str]:
        return ["spread_bps"]

    def compute(self, state: PipelineState, trade_idx: int) -> dict[str, float]:
        qt = int(state.trades["timestamp"].iloc[trade_idx])
        idx = state.last_bbo_idx(qt)
        if idx < 0:
            return {"spread_bps": np.nan}
        mid = state.bbo_mid[idx]
        spread = state.bbo_ask[idx] - state.bbo_bid[idx]
        return {"spread_bps": float(spread / mid * 10_000) if mid > 0 else np.nan}


class TradePressureFeature(FeatureBlock):
    """
    Aggressive buy/sell pressure proxy.
    Signed notional flow (buy - sell) over rolling windows, normalized by total flow.
    Also: taker imbalance = signed_flow / total_flow (scale-free).
    """

    def __init__(self, windows_s: tuple[float, ...] = (5.0, 30.0, 120.0)):
        self.windows_s = windows_s

    @property
    def output_columns(self) -> list[str]:
        cols = []
        for w in self.windows_s:
            cols += [f"signed_flow_{w}s", f"taker_imbalance_{w}s"]
        return cols

    def compute(self, state: PipelineState, trade_idx: int) -> dict[str, float]:
        qt = int(state.trades["timestamp"].iloc[trade_idx])
        out = {}
        for w in self.windows_s:
            sub = state.trade_window_slice(qt, w)
            if len(sub) == 0:
                out[f"signed_flow_{w}s"] = 0.0
                out[f"taker_imbalance_{w}s"] = 0.0
                continue
            s = np.where(sub["side"].str.lower() == "buy", 1.0, -1.0)
            notional = np.minimum(
                sub["price"].values * sub["amount"].values, NOTIONAL_CLIP
            )
            signed = (s * notional).sum()
            total = notional.sum()
            out[f"signed_flow_{w}s"] = float(signed)
            out[f"taker_imbalance_{w}s"] = float(signed / total) if total > 0 else 0.0
        return out


class LiquidationPressureFeature(FeatureBlock):
    """
    Forced/liquidation flow pressure proxy.
    Same-side and opposite-side liq notional over rolling windows.
    Normalized by NOTIONAL_CLIP for scale invariance.
    """

    def __init__(self, windows_s: tuple[float, ...] = (5.0, 30.0)):
        self.windows_s = windows_s

    @property
    def output_columns(self) -> list[str]:
        cols = []
        for w in self.windows_s:
            cols += [
                f"liq_buy_notional_{w}s",
                f"liq_sell_notional_{w}s",
                f"liq_same_side_{w}s",
                f"liq_opp_side_{w}s",
            ]
        return cols

    def compute(self, state: PipelineState, trade_idx: int) -> dict[str, float]:
        qt = int(state.trades["timestamp"].iloc[trade_idx])
        trade_side = str(state.trades["side"].iloc[trade_idx]).lower()
        out = {}

        for w in self.windows_s:
            buy_n, _ = state.liq_window_slice(qt, w, side="buy")
            sell_n, _ = state.liq_window_slice(qt, w, side="sell")

            buy_total = buy_n.sum() / NOTIONAL_CLIP if len(buy_n) > 0 else 0.0
            sell_total = sell_n.sum() / NOTIONAL_CLIP if len(sell_n) > 0 else 0.0

            # Same-side: liq in same direction as taker (maker takes opposite)
            same = sell_total if trade_side == "buy" else buy_total
            opp = buy_total if trade_side == "buy" else sell_total

            out[f"liq_buy_notional_{w}s"] = buy_total
            out[f"liq_sell_notional_{w}s"] = sell_total
            out[f"liq_same_side_{w}s"] = same
            out[f"liq_opp_side_{w}s"] = opp

        return out


class TimeFeature(FeatureBlock):
    """
    Time-of-day encoding (sin/cos for periodicity) and time to known events.
    Known scheduled events (UTC):
      - NYSE open:   14:30
      - NASDAQ open: 14:30
      - BTC funding: 00:00, 08:00, 16:00 (every 8h on Binance perps)
    """

    FUNDING_HOURS_UTC = (0, 8, 16)

    @property
    def output_columns(self) -> list[str]:
        return [
            "tod_sin",
            "tod_cos",  # time-of-day (24h cycle)
            "secs_to_nyse_open",  # seconds to next NYSE/NASDAQ open
            "secs_to_next_funding",  # seconds to next BTC funding
        ]

    def compute(self, state: PipelineState, trade_idx: int) -> dict[str, float]:
        qt_us = int(state.trades["timestamp"].iloc[trade_idx])
        dt = pd.Timestamp(qt_us * 1000, unit="ns", tz="UTC")

        secs_since_midnight = dt.hour * 3600 + dt.minute * 60 + dt.second
        total_secs = 86400.0
        tod_sin = np.sin(2 * np.pi * secs_since_midnight / total_secs)
        tod_cos = np.cos(2 * np.pi * secs_since_midnight / total_secs)

        # Seconds to next NYSE open (14:30 UTC on weekdays)
        nyse_open_secs = 14 * 3600 + 30 * 60
        if secs_since_midnight < nyse_open_secs:
            to_nyse = nyse_open_secs - secs_since_midnight
        else:
            to_nyse = (86400 - secs_since_midnight) + nyse_open_secs
        # Normalize: max 24h = 86400s → scale to [0, 1]
        to_nyse_norm = to_nyse / 86400.0

        # Seconds to next funding (every 8h: 00:00, 08:00, 16:00 UTC)
        funding_secs = [h * 3600 for h in self.FUNDING_HOURS_UTC]
        upcoming = [f for f in funding_secs if f > secs_since_midnight]
        if upcoming:
            to_funding = min(upcoming) - secs_since_midnight
        else:
            to_funding = (86400 - secs_since_midnight) + funding_secs[0]
        to_funding_norm = to_funding / 28800.0  # normalize by 8h period

        return {
            "tod_sin": float(tod_sin),
            "tod_cos": float(tod_cos),
            "secs_to_nyse_open": float(to_nyse_norm),
            "secs_to_next_funding": float(to_funding_norm),
        }


# ---------------------------------------------------------------------------
# 3. TRANSFORM LAYER
# ---------------------------------------------------------------------------


@dataclass
class Transform:
    """
    A declarative transform applied to a subset of feature columns.
    Supported ops: 'log', 'zscore', 'clamp', 'normalize'.
    """

    op: Literal["log", "zscore", "clamp", "normalize"]
    columns: list[str] | None = None  # None = apply to all numeric columns
    clip_range: tuple[float, float] = (-5.0, 5.0)  # for 'clamp'
    log_offset: float = 1e-8  # for 'log'

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cols = (
            self.columns
            if self.columns
            else df.select_dtypes(include=np.number).columns.tolist()
        )
        cols = [c for c in cols if c in df.columns]

        if self.op == "log":
            for c in cols:
                df[c] = np.log(np.abs(df[c]) + self.log_offset) * np.sign(df[c])
        elif self.op == "zscore":
            for c in cols:
                mu, sigma = df[c].mean(), df[c].std()
                if sigma > 1e-10:
                    df[c] = (df[c] - mu) / sigma
        elif self.op == "clamp":
            for c in cols:
                df[c] = df[c].clip(self.clip_range[0], self.clip_range[1])
        elif self.op == "normalize":
            for c in cols:
                mn, mx = df[c].min(), df[c].max()
                if mx > mn:
                    df[c] = (df[c] - mn) / (mx - mn)
        return df


class TransformLayer:
    """
    Applies a sequence of Transforms to the feature DataFrame.
    Transforms are applied in order (composable, declarative).
    """

    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        for t in self.transforms:
            df = t.apply(df)
        return df


# Default transform layer matching the spec
DEFAULT_TRANSFORMS = TransformLayer(
    [
        # Log-scale large flow features (signed_flow can be huge)
        Transform(
            op="log",
            columns=[
                c for window in (5.0, 30.0, 120.0) for c in [f"signed_flow_{window}s"]
            ],
        ),
        # Z-score price z-scores (they're already z-scores but re-normalize across dataset)
        Transform(
            op="zscore", columns=[f"price_zscore_{w}s" for w in (10.0, 60.0, 300.0)]
        ),
        # Clamp outliers in volatility and flow features
        Transform(
            op="clamp",
            columns=[f"realized_vol_{w}s" for w in (30.0, 300.0)]
            + [f"taker_imbalance_{w}s" for w in (5.0, 30.0, 120.0)],
            clip_range=(-3.0, 3.0),
        ),
        # Clamp liq features
        Transform(
            op="clamp",
            columns=[f"liq_same_side_{w}s" for w in (5.0, 30.0)]
            + [f"liq_opp_side_{w}s" for w in (5.0, 30.0)],
            clip_range=(0.0, 10.0),
        ),
    ]
)


# ---------------------------------------------------------------------------
# 4. SAMPLERS
# ---------------------------------------------------------------------------


class Sampler(ABC):
    """
    Determines WHICH trades emit a datapoint.
    Returns indices into the (sorted) trades DataFrame.
    """

    @abstractmethod
    def sample(self, trades: pd.DataFrame) -> list[int]:
        """Return list of trade indices to emit as datapoints."""
        ...


class VolumeClock(Sampler):
    """
    Emits a datapoint every time cumulative traded notional >= threshold.
    Classic volume-clock / volume-bar approach.
    """

    def __init__(self, threshold: float = 1_000_000.0):
        self.threshold = threshold

    def sample(self, trades: pd.DataFrame) -> list[int]:
        trades = trades.sort_values("timestamp").reset_index(drop=True)
        notional = (trades["price"] * trades["amount"]).values
        indices = []
        cumvol = 0.0
        for i, n in enumerate(notional):
            cumvol += n
            if cumvol >= self.threshold:
                indices.append(i)
                cumvol = 0.0
        return indices


class TimeClock(Sampler):
    """
    Emits a datapoint every `interval_s` seconds (time-bar approach).
    Picks the last trade in each time bucket.
    """

    def __init__(self, interval_s: float = 60.0):
        self.interval_s = interval_s

    def sample(self, trades: pd.DataFrame) -> list[int]:
        trades = trades.sort_values("timestamp").reset_index(drop=True)
        ts = trades["timestamp"].values.astype(np.int64)
        interval_us = int(self.interval_s * US_PER_SECOND)
        if len(ts) == 0:
            return []
        start = ts[0]
        end = ts[-1]
        indices = []
        bucket_end = start + interval_us
        while bucket_end <= end + interval_us:
            # Last trade before bucket_end
            hi = int(np.searchsorted(ts, bucket_end, side="left")) - 1
            lo = int(np.searchsorted(ts, bucket_end - interval_us, side="left"))
            if lo <= hi:
                indices.append(hi)
            bucket_end += interval_us
        return sorted(set(indices))


class TradeClock(Sampler):
    """
    Every trade is a datapoint (trade-bar approach).
    Optionally subsample by every `nth` trade.
    """

    def __init__(self, nth: int = 1):
        self.nth = nth

    def sample(self, trades: pd.DataFrame) -> list[int]:
        return list(range(0, len(trades), self.nth))


# ---------------------------------------------------------------------------
# 5. LABELER
# ---------------------------------------------------------------------------


class MarkoutLabeler:
    """
    Labels each sampled trade with the markout of the EXECUTED TRADE at tau seconds.

    Markout for a maker trade:
        s_i = +1 if taker buy (maker sold), -1 if taker sell (maker bought)
        mid_tau = forward mid at t_i + tau (last BBO with ts <= t_i + tau)
        markout_bps = -s_i * (mid_tau - price_i) / price_i * 10_000
        pnl_bps = markout_bps + maker_rebate_bps

    Returns NaN for trades where t_i + tau is beyond available BBO range.
    """

    MAKER_REBATE_BPS: float = 0.5

    def __init__(self, tau_s: float | list[float] = 30.0):
        self.taus = [tau_s] if isinstance(tau_s, (int, float)) else list(tau_s)

    def label(self, trades: pd.DataFrame, state: PipelineState) -> pd.DataFrame:
        """
        Add pnl_bps_{tau}s columns to trades DataFrame (subset = sampled indices).
        Also adds 's' and 'w' columns.
        """
        trades = trades.copy()
        ts_arr = trades["timestamp"].values.astype(np.int64)
        side_arr = np.where(trades["side"].str.lower() == "buy", 1.0, -1.0)
        price_arr = trades["price"].values.astype(np.float64)
        notional_arr = price_arr * trades["amount"].values.astype(np.float64)

        trades["s"] = side_arr
        trades["w"] = np.minimum(notional_arr, NOTIONAL_CLIP)

        bbo_ts = state.bbo_ts
        bbo_mid = state.bbo_mid
        max_bbo_ts = int(bbo_ts.max()) if len(bbo_ts) > 0 else 0

        for tau in self.taus:
            tau_us = int(tau * US_PER_SECOND)
            lookup_ts = ts_arr + tau_us
            idx = np.searchsorted(bbo_ts, lookup_ts, side="right") - 1
            edge_mask = lookup_ts > max_bbo_ts

            mid_tau = np.where(
                (idx >= 0) & (~edge_mask),
                bbo_mid[np.clip(idx, 0, len(bbo_mid) - 1)],
                np.nan,
            )

            markout = -side_arr * (mid_tau - price_arr) / price_arr * 10_000
            pnl = markout + self.MAKER_REBATE_BPS
            pnl = np.where(edge_mask, np.nan, pnl)
            trades[f"pnl_bps_{int(tau)}s"] = pnl

        return trades


# ---------------------------------------------------------------------------
# 6. DATASET BUILDER (orchestrator)
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """
    Declarative specification of the full pipeline.
    Change any field to alter behavior — no code changes needed.
    """

    # Feature blocks (in order)
    features: list[FeatureBlock] = field(
        default_factory=lambda: [
            TopOfBookVolumeFeature(),
            PriceZScoreFeature(windows_s=(10.0, 60.0, 300.0)),
            RollingVolatilityFeature(windows_s=(30.0, 300.0)),
            BidAskSpreadFeature(),
            TradePressureFeature(windows_s=(5.0, 30.0, 120.0)),
            LiquidationPressureFeature(windows_s=(5.0, 30.0)),
            TimeFeature(),
        ]
    )

    # Transform layer
    transforms: TransformLayer = field(default_factory=lambda: DEFAULT_TRANSFORMS)

    # Sampler
    sampler: Sampler = field(default_factory=lambda: VolumeClock(threshold=1_000_000.0))

    # Labeler
    labeler: MarkoutLabeler = field(
        default_factory=lambda: MarkoutLabeler(tau_s=[30.0, 120.0, 300.0])
    )

    # Whether to apply transforms (can disable for debugging)
    apply_transforms: bool = True


class DatasetBuilder:
    """
    Orchestrates the full pipeline:
        raw data → feature blocks → transforms → sampling → labeling → dataset
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def build(
        self,
        trades: pd.DataFrame,
        bbo: pd.DataFrame,
        liq_binance: pd.DataFrame | None = None,
        liq_bybit: pd.DataFrame | None = None,
        symbol: str = "unknown",
    ) -> pd.DataFrame:
        """
        Build the supervised dataset for one symbol period.

        Returns a DataFrame where each row is a sampled trade with:
          - feature columns (one per FeatureBlock output)
          - pnl_bps_{tau}s columns (targets)
          - metadata: timestamp, price, side, s, w
        """
        liq_binance = liq_binance if liq_binance is not None else pd.DataFrame()
        liq_bybit = liq_bybit if liq_bybit is not None else pd.DataFrame()

        # Build shared state
        state = PipelineState.build(trades, bbo, liq_binance, liq_bybit, symbol=symbol)

        logger.info(
            f"[{symbol}] Building dataset: {len(trades):,} trades, "
            f"sampler={self.config.sampler.__class__.__name__}"
        )

        # Sampling
        sample_indices = self.config.sampler.sample(state.trades)
        logger.info(f"[{symbol}] Sampled {len(sample_indices):,} datapoints")

        if not sample_indices:
            return pd.DataFrame()

        # Compute features for each sampled trade
        rows = []
        for idx in sample_indices:
            row: dict[str, Any] = {}
            for block in self.config.features:
                row.update(block.compute(state, idx))
            rows.append(row)

        feat_df = pd.DataFrame(rows)
        feat_df.index = pd.Index([state.trades.index[i] for i in sample_indices])

        # Apply transforms
        if self.config.apply_transforms:
            feat_df = self.config.transforms.apply(feat_df)

        # Labeling (markout of executed trade)
        sampled_trades = state.trades.iloc[sample_indices].reset_index(drop=True)
        labeled = self.config.labeler.label(sampled_trades, state)

        # Merge: metadata + features + labels
        meta_cols = ["timestamp", "price", "side", "amount", "s", "w"]
        meta_cols = [c for c in meta_cols if c in labeled.columns]
        target_cols = [c for c in labeled.columns if c.startswith("pnl_bps_")]

        dataset = labeled[meta_cols + target_cols].reset_index(drop=True)
        feat_df_reset = feat_df.reset_index(drop=True)
        dataset = pd.concat([dataset, feat_df_reset], axis=1)

        # Drop rows with all-NaN targets
        n_before = len(dataset)
        dataset = dataset.dropna(subset=target_cols, how="all").reset_index(drop=True)
        logger.info(
            f"[{symbol}] Dataset: {n_before} → {len(dataset)} rows after NaN drop"
        )

        return dataset

    def feature_names(self) -> list[str]:
        """Return all feature column names (in order)."""
        cols = []
        for block in self.config.features:
            cols.extend(block.output_columns)
        return cols

    def target_names(self) -> list[str]:
        return [f"pnl_bps_{int(t)}s" for t in self.config.labeler.taus]


# ---------------------------------------------------------------------------
# 7. MODEL TRAINING INTERFACE (decoupled from features)
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """
    Declarative model specification.
    """

    model_type: Literal["lgbm", "linear", "xgb"] = "lgbm"
    target_col: str = "pnl_bps_30s"
    use_sample_weights: bool = True  # weight by w_i
    params: dict[str, Any] = field(default_factory=dict)


def train_model(
    dataset: pd.DataFrame,
    feature_names: list[str],
    spec: ModelSpec,
) -> Any:
    """
    Clean separation: train a model given features, target, and spec.
    No feature logic here — only model fitting.

    Parameters
    ----------
    dataset : pd.DataFrame
        Output of DatasetBuilder.build()
    feature_names : list[str]
        Column names to use as features (from DatasetBuilder.feature_names())
    spec : ModelSpec
        Model configuration

    Returns
    -------
    Trained model object with .predict(X) method.
    """
    feat_cols = [c for c in feature_names if c in dataset.columns]
    X = dataset[feat_cols].fillna(0.0)
    y = dataset[spec.target_col].fillna(0.0)
    w = (
        dataset["w"].fillna(1.0)
        if spec.use_sample_weights and "w" in dataset.columns
        else None
    )

    # Drop rows where target is NaN
    valid = ~dataset[spec.target_col].isna()
    X, y = X[valid], y[valid]
    if w is not None:
        w = w[valid]

    logger.info(
        f"Training {spec.model_type} on {len(X):,} samples, "
        f"{len(feat_cols)} features, target={spec.target_col}"
    )

    if spec.model_type == "lgbm":
        from lightgbm import LGBMRegressor

        default_params = dict(
            objective="huber",
            alpha=0.9,
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=16,
            min_child_samples=100,
            subsample=0.6,
            colsample_bytree=0.6,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        default_params.update(spec.params)
        model = LGBMRegressor(**default_params)
        model.fit(X, y, sample_weight=w)

    elif spec.model_type == "linear":
        from sklearn.linear_model import Ridge

        params = {"alpha": 1.0, **spec.params}
        model = Ridge(**params)
        model.fit(X, y, sample_weight=w)

    elif spec.model_type == "xgb":
        import xgboost as xgb

        default_params = dict(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
        default_params.update(spec.params)
        model = xgb.XGBRegressor(**default_params)
        fit_kwargs = {"sample_weight": w} if w is not None else {}
        model.fit(X, y, **fit_kwargs)

    else:
        raise ValueError(f"Unknown model_type: {spec.model_type}")

    return model


def evaluate_model(
    model: Any,
    dataset: pd.DataFrame,
    feature_names: list[str],
    spec: ModelSpec,
) -> dict[str, float]:
    """
    Compute evaluation metrics on a dataset split.
    Returns dict of metric name → value.
    """
    feat_cols = [c for c in feature_names if c in dataset.columns]
    X = dataset[feat_cols].fillna(0.0)
    y = dataset[spec.target_col]
    w = dataset["w"] if "w" in dataset.columns else None

    valid = ~y.isna()
    X, y = X[valid], y[valid]
    if w is not None:
        w = w[valid]

    preds = model.predict(X)

    # Weighted correlation (main metric for markout prediction)
    if w is not None and w.sum() > 0:
        wn = w.values / w.sum()
        wmean_y = (wn * y.values).sum()
        wmean_p = (wn * preds).sum()
        wcov = (wn * (y.values - wmean_y) * (preds - wmean_p)).sum()
        wstd_y = np.sqrt((wn * (y.values - wmean_y) ** 2).sum())
        wstd_p = np.sqrt((wn * (preds - wmean_p) ** 2).sum())
        wcorr = wcov / (wstd_y * wstd_p + 1e-10)
    else:
        wcorr = np.corrcoef(y.values, preds)[0, 1]

    # WMSE
    if w is not None:
        wmse = float((w.values * (y.values - preds) ** 2).sum() / w.sum())
    else:
        wmse = float(((y.values - preds) ** 2).mean())

    # IC (information coefficient = Spearman rank correlation)
    from scipy import stats

    ic, _ = stats.spearmanr(y.values, preds)

    return {
        "weighted_corr": float(wcorr),
        "ic_spearman": float(ic),
        "wmse": float(wmse),
        "n_samples": int(valid.sum()),
        "target": spec.target_col,
    }
