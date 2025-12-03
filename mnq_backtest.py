import math
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# =========================
# Config / parameters
# =========================

TICKER = "MNQ=F"
INTERVAL = "1m"
PERIOD = "7d"        # 1m supports ~7d; we resample to higher TFs for trend logic

# Trend / momentum (some legacy, some still used)
L_TREND = 32          # kept but not central now
R_TREND_MIN = 0.003
EMA_FAST = 50
EMA_SLOW = 200
EMA_PULL = 20

# Intraday trend filter and risk params
EMA20_1M = 20
EMA50_1M = 50
ATR_PERIOD = 14
ATR_MULT = 1.25              # legacy; ATR-based stops now use 2x ATR
MIN_POINTS_RISK = 5.0        # legacy
MAX_POINTS_RISK = 8.0        # legacy
BREAKEVEN_TRIGGER_POINTS = 12.0
TRAIL1_TRIGGER = 18.0        # no longer used (replaced by ATR trailing)
TRAIL1_POINTS = 8.0
TRAIL2_TRIGGER = 30.0
TRAIL2_POINTS = 12.0
TAKE_PARTIAL = True
PARTIAL_AT_POINTS = 15.0
PARTIAL_PCT = 0.5
MAX_TRADES_PER_DAY = 5            # kept only for diagnostics
MIN_BARS_BE = 2                   # wait at least 2 bars before moving to breakeven
MAX_TRADES_PER_DAY_STRONG = 8     # kept only for diagnostics
ATR_TRAIL1_MULT = 1.25            # legacy
ATR_TRAIL2_MULT = 1.0             # legacy

# OU-style pullback (still computed, but not used in entry now)
N_DEV = 40
Z_ENTRY = 1.0
Z_EXIT = 0.1

# Swing high/low lookback for structural stop (not central any more)
M_SWING = 3
STOP_BUFFER_POINTS = 0.75

# Risk
MULTIPLIER = 2.0           # MNQ $2 per index point per contract
MAX_CONTRACTS = 10         # cap MNQ size for safer trend-following
MIN_QTY = 1                # minimum contracts per entry (risk-based floor)
RISK_PER_TRADE = 80.0      # target dollar risk per trade
MAX_TRADE_RISK = 500.0     # legacy
DAILY_LOSS_LIMIT = -250.0  # hard daily loss cap
 
# Optional: force a fixed contract quantity per trade (overrides risk-based sizing)
# Set to an integer (e.g., 2, 4) to enable; leave as None to use risk-based sizing (with MIN_QTY floor)
FIXED_QTY: Optional[int] = None  # Ignored for live/backtest sizing; always use risk-based qty

# Time filters (Central Time)
# Trading sessions: allow NY, London, Asia; block 3:00–5:00 PM Central
SESSION_START_HOUR = 7
SESSION_START_MIN = 0
SESSION_END_ENTRY_HOUR = 15
SESSION_END_ENTRY_MIN = 10
FORCE_FLAT_HOUR = 15
FORCE_FLAT_MIN = 30
NO_TRADE_START_HOUR = 15
NO_TRADE_START_MIN = 0
NO_TRADE_END_HOUR = 17
NO_TRADE_END_MIN = 0
REOPEN_SKIP_MINUTES = 5  # skip first minutes after 5pm CT reopen

# Time stop (bars)
# Live-aligned exit settings for backtest
# These mirror defaults in live_trade_topstep.py
ATR_STOP_MULT = 1.25
BE_TRIGGER_ATR = 1.5
BE_OFFSET_ATR = 0.10
TRAIL1_TRIGGER_ATR = 2.5
TRAIL1_MULT = 2.0
TRAIL2_TRIGGER_ATR = 4.0
TRAIL2_MULT = 1.5
TRAIL3_TRIGGER_ATR = 5.0
TRAIL3_MULT = 1.25
TIME_STOP_MIN = 90  # minutes
TIME_STOP_EXT_MIN = 60  # extend time stop by 60 minutes if progress >= 1x ATR
EXHAUSTION_ATR_MULT = 4.0  # overshoot threshold vs hourly EMA (backtest-only)

# Pyramiding (adds on favorable movement)
PYRAMID_ENABLED = True
PYRAMID_STEP_ATR = 1.5
PYRAMID_ADD_QTY = 1

STARTING_CAPITAL = 100_000.0

# Donchian settings (15m trend-following entries)
DONCHIAN_LEN = 10

# Per-ticker point values (for $ATR filtering)
PNT_VALUE = {
    "MNQ": 2.0,
    "MES": 5.0,
    "MYM": 0.5,
    "M2K": 5.0,
}

def point_value_for(symbol: str) -> float:
    return float(PNT_VALUE.get(symbol.upper(), MULTIPLIER))

# Per-ticker tick sizes for stop buffer parity with live
TICK_SIZE_MAP = {
    "MNQ": 0.25,
    "MES": 0.25,
    "MYM": 1.0,
    "M2K": 0.1,
}

def tick_size_for(symbol: str) -> float:
    return float(TICK_SIZE_MAP.get(symbol.upper(), 0.25))

def is_mnq_symbol(symbol: Optional[str]) -> bool:
    if not symbol:
        return False
    return "MNQ" in str(symbol).upper()

def atr_in_band_for_symbol(symbol: str, atr_points: float) -> bool:
    try:
        atr_points = float(atr_points)
    except Exception:
        return False
    if is_mnq_symbol(symbol):
        return (atr_points >= MNQ_ATR_MIN) and (atr_points <= MNQ_ATR_MAX)
    # Non-MNQ: fall back to global $ATR band
    pv = point_value_for(symbol)
    datr = atr_points * pv
    return (datr >= DATR_MIN_GLOBAL) and (datr <= DATR_MAX_GLOBAL)

def is_overextended(row) -> bool:
    """Apply overextension filter only when trend is sufficiently strong (15m ADX high)."""
    try:
        adx15 = float(row.get("adx_15m") or np.nan)
    except Exception:
        adx15 = np.nan
    # Only block when ADX is strong enough to consider extension meaningful
    try:
        adx_thr = ADX_TREND_MIN
    except Exception:
        adx_thr = 20.0
    if np.isnan(adx15) or adx15 < adx_thr:
        return False
    try:
        close = float(row.get("Close") if "Close" in row else row.get("close"))
        ema_1h = row.get("ema_1h_50") if "ema_1h_50" in row else row.get("ema_1h")
        atr = float(row.get("atr") or row.get("atr14") or 0.0)
        if ema_1h is None or atr <= 0:
            return False
        ema_1h = float(ema_1h)
        return abs(close - ema_1h) > OVEREXT_ATR_MULT * atr
    except Exception:
        return False

def micro_pullback_ok(df: pd.DataFrame, ts: pd.Timestamp, direction: str) -> bool:
    if not REQ_MICRO_PULLBACK:
        return True
    if ts not in df.index:
        return True
    hist = df.loc[:ts].iloc[:-1].tail(PULLBACK_LOOKBACK)
    if hist.empty:
        return False
    try:
        current_close = float(df.loc[ts, "Close"])
    except Exception:
        return True
    if direction == "LONG":
        return (current_close - hist["Close"]).max() >= PULLBACK_MIN_POINTS
    if direction == "SHORT":
        return (hist["Close"] - current_close).max() >= PULLBACK_MIN_POINTS
    return True


def atr_in_mr_band(atr_val: float) -> bool:
    try:
        v = float(atr_val)
    except Exception:
        return False
    return (v >= MR_ATR_MIN) and (v <= MR_ATR_MAX)


def get_mr_z(row) -> Optional[float]:
    try:
        if MR_Z_COL in row:
            return float(row.get(MR_Z_COL))
    except Exception:
        pass
    # common fallbacks
    for alt in ("Z", "zscore_dev", "z_dev", "dev_z"):
        try:
            if alt in row:
                return float(row.get(alt))
        except Exception:
            continue
    return None

def _parse_band_map(raw: str):
    m = {}
    for seg in raw.split(";"):
        seg = seg.strip()
        if not seg or ":" not in seg or "-" not in seg:
            continue
        sym, rng = seg.split(":", 1)
        sym = sym.strip().upper()
        try:
            lo_s, hi_s = rng.split("-", 1)
            lo = float(lo_s.strip())
            hi = float(hi_s.strip())
            if lo > 0 and hi > 0 and hi >= lo:
                m[sym] = (lo, hi)
        except Exception:
            pass
    return m

ATR_BAND_MAP = _parse_band_map(os.environ.get("TOPSTEP_ATR_ENTRY_BANDS", ""))
# Built-in ATR defaults (points)
DEFAULT_ATR_BANDS = {
    "MNQ": (10.0, 20.0),
}
# Lower, sensible default $ATR bands if none provided via env
DEFAULT_DATR_BANDS = {
    "MNQ": (6.0, 30.0),
    "MES": (5.0, 25.0),
    "MYM": (2.0, 10.0),
    "M2K": (1.0, 10.0),
}
_DATR_ENV = _parse_band_map(os.environ.get("TOPSTEP_DATR_ENTRY_BANDS", ""))
DATR_BAND_MAP = {**DEFAULT_DATR_BANDS, **_DATR_ENV}
BASE_SYMBOL = (TICKER.split("=")[0] if "=" in TICKER else TICKER).strip().upper()

# Disable ATR/$ATR entry gating for backtest (still requires ATR > 0)
ATR_FILTER_ENABLED = False

# Stop tightening hysteresis (in ATRs), close-confirmation count, and tick buffer (ticks)
MIN_STOP_STEP_ATR = 0.25
STOP_CLOSE_CONFIRMED = 1
STOP_TICK_BUFFER = 2.0

# Global Dollar-ATR entry band (normalized to MYM behavior)
try:
    DATR_MIN_GLOBAL = float(os.environ.get("TOPSTEP_GLOBAL_DATR_MIN", "5").strip())
except Exception:
    DATR_MIN_GLOBAL = 5.0
try:
    DATR_MAX_GLOBAL = float(os.environ.get("TOPSTEP_GLOBAL_DATR_MAX", "20").strip())
except Exception:
    DATR_MAX_GLOBAL = 20.0

# Regime classifier thresholds (env-based)
try:
    HL_MR_MAX = float(os.environ.get("HL_MR_MAX", "5.0").strip())
except Exception:
    HL_MR_MAX = 5.0
try:
    HL_TREND_MIN = float(os.environ.get("HL_TREND_MIN", "30.0").strip())
except Exception:
    HL_TREND_MIN = 30.0
try:
    ADX_MR_MAX = float(os.environ.get("ADX_MR_MAX", "15.0").strip())
except Exception:
    ADX_MR_MAX = 15.0
try:
    ADX_TREND_MIN = float(os.environ.get("ADX_TREND_MIN", "20.0").strip())
except Exception:
    ADX_TREND_MIN = 20.0
try:
    BB_COMPRESS_RATIO = float(os.environ.get("BB_COMPRESS_RATIO", "0.8").strip())
except Exception:
    BB_COMPRESS_RATIO = 0.8
try:
    BB_EXPAND_RATIO = float(os.environ.get("BB_EXPAND_RATIO", "1.2").strip())
except Exception:
    BB_EXPAND_RATIO = 1.2

# Compatibility: allow alternate env names to override the same thresholds
def _override_from_alt_env():
    global ADX_TREND_MIN, ADX_MR_MAX, BB_EXPAND_RATIO, BB_COMPRESS_RATIO, HL_MR_MAX
    try:
        v = os.environ.get("ADX_TREND")
        if v:
            ADX_TREND_MIN = float(v)
    except Exception:
        pass
    try:
        v = os.environ.get("ADX_MR")
        if v:
            ADX_MR_MAX = float(v)
    except Exception:
        pass
    try:
        v = os.environ.get("BB_EXPAND_MULT")
        if v:
            BB_EXPAND_RATIO = float(v)
    except Exception:
        pass
    try:
        v = os.environ.get("BB_SQUEEZE_MULT")
        if v:
            BB_COMPRESS_RATIO = float(v)
    except Exception:
        pass
    try:
        v = os.environ.get("MR_MAX_HALF_LIFE")
        if v:
            HL_MR_MAX = float(v)
    except Exception:
        pass

_override_from_alt_env()

# Winner protection safety rule
try:
    SAFE_MFE_POINTS = float(os.environ.get("SAFE_MFE_POINTS", "6.0").strip())
except Exception:
    SAFE_MFE_POINTS = 6.0
try:
    SAFE_BE_OFFSET_POINTS = float(os.environ.get("SAFE_BE_OFFSET_POINTS", "0.5").strip())
except Exception:
    SAFE_BE_OFFSET_POINTS = 0.5

# MNQ-specific ATR band (points)
try:
    MNQ_ATR_MIN = float(os.environ.get("MNQ_ATR_MIN", "2.0").strip())
except Exception:
    MNQ_ATR_MIN = 2.0
try:
    MNQ_ATR_MAX = float(os.environ.get("MNQ_ATR_MAX", "20.0").strip())
except Exception:
    MNQ_ATR_MAX = 20.0

# Overextension filter threshold vs 1h EMA
try:
    OVEREXT_ATR_MULT = float(os.environ.get("OVEREXT_ATR_MULT", "3.0").strip())
except Exception:
    OVEREXT_ATR_MULT = 3.0

# MNQ-tuned BE/Trail defaults (overrides earlier literals)
try:
    BE_TRIGGER_ATR = float(os.environ.get("BE_TRIGGER_ATR", "1.0").strip())
except Exception:
    BE_TRIGGER_ATR = 1.0
try:
    BE_OFFSET_ATR = float(os.environ.get("BE_OFFSET_ATR", "0.25").strip())
except Exception:
    BE_OFFSET_ATR = 0.25
try:
    TRAIL1_TRIGGER_ATR = float(os.environ.get("TRAIL1_TRIGGER_ATR", "1.5").strip())
except Exception:
    TRAIL1_TRIGGER_ATR = 1.5
try:
    TRAIL1_MULT = float(os.environ.get("TRAIL1_MULT", "1.0").strip())
except Exception:
    TRAIL1_MULT = 1.0

# Optional micro-pullback before breakout entries
REQ_MICRO_PULLBACK = (os.environ.get("REQ_MICRO_PULLBACK", "0").strip().lower() not in ("0", "false"))
try:
    PULLBACK_LOOKBACK = int(float(os.environ.get("PULLBACK_LOOKBACK", "5").strip()))
except Exception:
    PULLBACK_LOOKBACK = 5
try:
    PULLBACK_MIN_POINTS = float(os.environ.get("PULLBACK_MIN_POINTS", "3.0").strip())
except Exception:
    PULLBACK_MIN_POINTS = 3.0

# Mean-reversion (MR) strategy knobs
MR_ENABLE = (os.environ.get("MR_ENABLE", "1").strip().lower() not in ("0", "false"))
MR_Z_COL = os.environ.get("MR_Z_COL", "mr_z").strip() or "mr_z"
try:
    MR_Z_ENTRY = float(os.environ.get("MR_Z_ENTRY", "0.9").strip())
except Exception:
    MR_Z_ENTRY = 0.9
try:
    MR_Z_TP_ZONE = float(os.environ.get("MR_Z_TP_ZONE", "0.35").strip())
except Exception:
    MR_Z_TP_ZONE = 0.35
try:
    MR_Z_MAX_EXTREME = float(os.environ.get("MR_Z_MAX_EXTREME", "3.0").strip())
except Exception:
    MR_Z_MAX_EXTREME = 3.0
try:
    MR_ATR_STOP_MULT = float(os.environ.get("MR_ATR_STOP_MULT", "1.0").strip())
except Exception:
    MR_ATR_STOP_MULT = 1.0
try:
    MR_BE_TRIGGER_ATR = float(os.environ.get("MR_BE_TRIGGER_ATR", "0.75").strip())
except Exception:
    MR_BE_TRIGGER_ATR = 0.75
try:
    MR_BE_OFFSET_ATR = float(os.environ.get("MR_BE_OFFSET_ATR", "0.10").strip())
except Exception:
    MR_BE_OFFSET_ATR = 0.10
try:
    MR_Z_EXIT = float(os.environ.get("MR_Z_EXIT", "0.2").strip())
except Exception:
    MR_Z_EXIT = 0.2
try:
    MR_ATR_MIN = float(os.environ.get("MR_ATR_MIN", "3.0").strip())
except Exception:
    MR_ATR_MIN = 3.0
try:
    MR_ATR_MAX = float(os.environ.get("MR_ATR_MAX", "16.0").strip())
except Exception:
    MR_ATR_MAX = 16.0


@dataclass
class Trade:
    direction: str       # "LONG" or "SHORT"
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: int
    pnl: float
    reason: str
    # Added for logging/analysis
    initial_stop: Optional[float] = None     # stop price at entry, based on ATR
    atr_at_entry: Optional[float] = None     # ATR value at entry
    stop_points_entry: Optional[float] = None  # 2x ATR distance (points) used for initial stop
    mode: Optional[str] = None               # 'TREND' or 'MR'
    # New diagnostics
    is_full_exit: bool = True                # False for partial records
    mfe_points: Optional[float] = None       # max favorable excursion in points
    mfe_price_at_exit: Optional[float] = None
    entry_qty: Optional[int] = None         # size at entry
    max_qty: Optional[int] = None           # max size reached (with pyramiding)
    # Strategy annotations
    strategy: Optional[str] = None          # 'TREND' or 'MR'
    entry_reason: Optional[str] = None      # e.g., 'BREAKOUT' or 'MR_Z_REVERSION'


# =========================
# Data loading & indicators
# =========================

def load_mnq_data(interval: Optional[str] = None, period: Optional[str] = None, ticker: Optional[str] = None):
    # Ensure columns come grouped by OHLCV field (not by ticker) and keep
    # unadjusted prices to maintain prior behavior and avoid FutureWarning.
    iv = interval or INTERVAL
    pr = period or PERIOD
    tk = ticker or TICKER

    def _download(interval: str, period: str):
        return yf.download(
            tk,
            interval=interval,
            period=period,
            group_by="column",
            auto_adjust=False,
        )

    df = _download(iv, pr)

    if df.empty:
        # Yahoo blocks >60d history for sub-hour intraday intervals. Fallback to hourly.
        fallback_interval = None
        if isinstance(iv, str) and iv.endswith("m") and pr not in ("1d", "5d", "7d", "30d", "60d"):
            fallback_interval = "60m"
            df = _download(fallback_interval, pr)

        if df.empty:
            raise RuntimeError("No data downloaded. Check ticker/interval/period.")

    # yfinance index is usually tz-aware UTC; convert to US/Central
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("US/Central")
    else:
        df.index = df.index.tz_convert("US/Central")

    # If yfinance returns MultiIndex columns (e.g., field x ticker), flatten to fields.
    if isinstance(df.columns, pd.MultiIndex):
        # If the first level looks like fields, drop the ticker level.
        if "Close" in df.columns.get_level_values(0):
            df = df.droplevel(1, axis=1)
        # Otherwise, try selecting by ticker then use fields.
        elif tk in df.columns.get_level_values(0):
            df = df.xs(tk, level=0, axis=1)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # EMAs (1m base)
    df["ema_fast"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["ema_pull"] = df["Close"].ewm(span=EMA_PULL, adjust=False).mean()

    # ATR(14) on base timeframe for stops
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(ATR_PERIOD).mean()
    df["atr"] = df["atr14"]
    df["atr_pct"] = df["atr"] / df["Close"]

    # Intraday VWAP on base TF (resets daily)
    day = df.index.tz_convert("US/Central").date if df.index.tz is not None else df.index.date
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    df["cum_pv"] = (typical * df["Volume"]).groupby(day).cumsum()
    df["cum_v"] = df["Volume"].groupby(day).cumsum()
    df["vwap"] = df["cum_pv"] / df["cum_v"]
    df["vol_ma50"] = df["Volume"].rolling(50).mean()

    # 1-minute EMAs for local intraday trend (kept, may be useful)
    df["ema20_1"] = df["Close"].ewm(span=EMA20_1M, adjust=False).mean()
    df["ema50_1"] = df["Close"].ewm(span=EMA50_1M, adjust=False).mean()

    # Build 5-minute resample for some diagnostics (ADX, squeeze) and align to base TF
    ohlc_5 = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    df5 = df[["Open", "High", "Low", "Close", "Volume"]].resample("5min").agg(ohlc_5).dropna()
    df5["ema20_5"] = df5["Close"].ewm(span=20, adjust=False).mean()
    df5["ema50_5"] = df5["Close"].ewm(span=50, adjust=False).mean()
    # ADX(14) on 5m for diagnostics
    tr1_5 = df5["High"] - df5["Low"]
    tr2_5 = (df5["High"] - df5["Close"].shift(1)).abs()
    tr3_5 = (df5["Low"] - df5["Close"].shift(1)).abs()
    tr_5 = pd.concat([tr1_5, tr2_5, tr3_5], axis=1).max(axis=1)
    plus_dm_5 = np.maximum(df5["High"] - df5["High"].shift(1), 0)
    minus_dm_5 = np.maximum(df5["Low"].shift(1) - df5["Low"], 0)
    tr14_5 = tr_5.rolling(14).sum()
    plus14_5 = plus_dm_5.rolling(14).sum()
    minus14_5 = minus_dm_5.rolling(14).sum()
    plus_di_5 = 100 * plus14_5 / tr14_5
    minus_di_5 = 100 * minus14_5 / tr14_5
    dx_5 = 100 * (plus_di_5 - minus_di_5).abs() / (plus_di_5 + minus_di_5)
    df5["adx14_5"] = dx_5.rolling(14).mean()
    # 5m Bollinger Bandwidth (squeeze) using SMA20 and std20
    df5["sma20_5"] = df5["Close"].rolling(20).mean()
    std20_5 = df5["Close"].rolling(20).std()
    upper_5 = df5["sma20_5"] + 2.0 * std20_5
    lower_5 = df5["sma20_5"] - 2.0 * std20_5
    df5["bb_bw_5"] = (upper_5 - lower_5) / df5["sma20_5"].abs()
    df5["bb_bw_ma_5"] = df5["bb_bw_5"].rolling(50).mean()
    bias = df5[["ema20_5", "ema50_5"]].reindex(df.index, method="ffill")
    df[["ema20_5", "ema50_5"]] = bias
    df["adx14_5"] = df5["adx14_5"].reindex(df.index, method="ffill")
    df["bb_bw_5"] = df5["bb_bw_5"].reindex(df.index, method="ffill")
    df["bb_bw_ma_5"] = df5["bb_bw_ma_5"].reindex(df.index, method="ffill")

    # Build 15-minute ADX for regime classification
    ohlc_15 = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    df15 = df[["Open", "High", "Low", "Close", "Volume"]].resample("15min").agg(ohlc_15).dropna()
    tr1_15 = df15["High"] - df15["Low"]
    tr2_15 = (df15["High"] - df15["Close"].shift(1)).abs()
    tr3_15 = (df15["Low"] - df15["Close"].shift(1)).abs()
    tr_15 = pd.concat([tr1_15, tr2_15, tr3_15], axis=1).max(axis=1)
    plus_dm_15 = np.maximum(df15["High"] - df15["High"].shift(1), 0)
    minus_dm_15 = np.maximum(df15["Low"].shift(1) - df15["Low"], 0)
    tr14_15 = tr_15.rolling(14).sum()
    plus14_15 = plus_dm_15.rolling(14).sum()
    minus14_15 = minus_dm_15.rolling(14).sum()
    plus_di_15 = 100 * plus14_15 / tr14_15
    minus_di_15 = 100 * minus14_15 / tr14_15
    dx_15 = 100 * (plus_di_15 - minus_di_15).abs() / (plus_di_15 + minus_di_15)
    df15["adx_15m"] = dx_15.rolling(14).mean()
    df["adx_15m"] = df15["adx_15m"].reindex(df.index, method="ffill")

    # Legacy 1m breakout levels (not used for entry any more)
    N_BREAK = 20
    df["break_high"] = df["High"].rolling(N_BREAK).max().shift(1)
    df["break_low"] = df["Low"].rolling(N_BREAK).min().shift(1)

    # Simple moving averages and trend signals with slope (kept for diagnostics)
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()
    df["ma_slope"] = df["ma20"].diff()
    df["trend_up"] = (df["ma20"] > df["ma50"]) & (df["ma_slope"] > 0)
    df["trend_down"] = (df["ma20"] < df["ma50"]) & (df["ma_slope"] < 0)

    # Compute ADX 14 on base TF (diagnostic)
    tr1_adx = df["High"] - df["Low"]
    tr2_adx = (df["High"] - df["Close"].shift(1)).abs()
    tr3_adx = (df["Low"] - df["Close"].shift(1)).abs()
    tr_adx = pd.concat([tr1_adx, tr2_adx, tr3_adx], axis=1).max(axis=1)

    plus_dm = np.maximum(df["High"] - df["High"].shift(1), 0)
    minus_dm = np.maximum(df["Low"].shift(1) - df["Low"], 0)

    tr14 = tr_adx.rolling(14).sum()
    plus14 = plus_dm.rolling(14).sum()
    minus14 = minus_dm.rolling(14).sum()

    plus_di = 100 * plus14 / tr14
    minus_di = 100 * minus14 / tr14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df["adx14"] = dx.rolling(14).mean()

    # time-series momentum return over L_TREND bars (still computed)
    df["ret_L"] = df["Close"] / df["Close"].shift(L_TREND) - 1.0

    # deviation & Z-score (OU flavor, still computed)
    df["y"] = pd.Series(df["Close"]).astype(float) - pd.Series(df["ema_pull"]).astype(float)
    df["y_mean"] = df["y"].rolling(N_DEV).mean()
    df["y_std"] = df["y"].rolling(N_DEV).std()
    df["Z"] = (df["y"] - df["y_mean"]) / df["y_std"]

    # OU half-life (rolling estimate)
    window = 200
    hl_list = [np.nan] * len(df)
    for i in range(window, len(df)):
        y_win = df["y"].iloc[i - window: i]
        dy_win = y_win.diff().dropna()
        y_lag_win = y_win.shift(1).dropna()
        if len(dy_win) != len(y_lag_win):
            continue
        lambda_est = np.polyfit(y_lag_win.values, dy_win.values, 1)[0]
        if lambda_est < 0:
            hl_list[i] = -np.log(2.0) / lambda_est
        else:
            hl_list[i] = np.nan
    df["half_life"] = hl_list
    df.loc[df["half_life"] > 200, "half_life"] = 200

    # ================
    # 1H trend context
    # ================
    ohlc_1h = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    df_1h = df[["Open", "High", "Low", "Close", "Volume"]].resample("1h").agg(ohlc_1h).dropna()
    df_1h["ema_1h_50"] = df_1h["Close"].ewm(span=50, adjust=False).mean()
    df_1h["ema_1h_slope"] = df_1h["ema_1h_50"].diff()
    df[["ema_1h_50", "ema_1h_slope"]] = df_1h[["ema_1h_50", "ema_1h_slope"]].reindex(df.index, method="ffill")

    # ================
    # Donchian breakout levels for trend entries (base timeframe)
    # ================
    df["donch_high"] = df["High"].rolling(DONCHIAN_LEN).max().shift(1)
    df["donch_low"] = df["Low"].rolling(DONCHIAN_LEN).min().shift(1)

    return df.dropna().copy()


def classify_regime(row) -> str:
    """
    Determine trend regime using hourly EMA(50) and its slope.
    This is the high-level trend filter.
    """
    ema_1h = row.get("ema_1h_50", np.nan)
    slope_1h = row.get("ema_1h_slope", np.nan)
    close = row["Close"]

    if np.isnan(ema_1h) or np.isnan(slope_1h):
        return "NONE"

    if close > ema_1h and slope_1h > 0:
        return "UP"
    elif close < ema_1h and slope_1h < 0:
        return "DOWN"
    else:
        return "NONE"


def in_entry_session(ts: pd.Timestamp) -> bool:
    """Allow entries only in specific liquid windows (US/Central):
    - 08:30–11:00 (US open)
    - 12:45–15:00 (US afternoon)
    - 02:00–04:00 (London open)
    (Asian session 19:00–22:00 is excluded for backtest entries)
    """
    t = ts

    def in_window(h1: int, m1: int, h2: int, m2: int) -> bool:
        start = t.replace(hour=h1, minute=m1, second=0, microsecond=0)
        end = t.replace(hour=h2, minute=m2, second=0, microsecond=0)
        return start <= t <= end

    return (
        in_window(8, 30, 11, 0) or
        in_window(12, 45, 15, 0) or
        in_window(2, 0, 4, 0)
    )


def past_force_flat(ts: pd.Timestamp) -> bool:
    """Force flat during 3:00–5:00 PM Central maintenance window to avoid holding through the break."""
    t = ts
    block_start = t.replace(
        hour=NO_TRADE_START_HOUR,
        minute=NO_TRADE_START_MIN,
        second=0,
        microsecond=0,
    )
    block_end = t.replace(
        hour=NO_TRADE_END_HOUR,
        minute=NO_TRADE_END_MIN,
        second=0,
        microsecond=0,
    )
    return block_start <= t < block_end


# =========================
# Backtest engine
# =========================

def backtest(df: pd.DataFrame, base_symbol: Optional[str] = None):
    position = 0  # +qty for long, -qty for short
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None  # not heavily used (trail dominates)
    entry_index: Optional[int] = None
    atr_at_entry: Optional[float] = None
    stop_points_entry: Optional[float] = None
    entry_time: Optional[pd.Timestamp] = None
    mfe_price: Optional[float] = None  # best favorable price since entry (high for long, low for short)

    realized_pnl = 0.0
    daily_pnl = 0.0
    trading_enabled = True
    peak_equity = STARTING_CAPITAL
    MAX_DD_LIMIT = 2000.0  # account liquidation threshold

    trades: List[Trade] = []
    equity_curve = []
    timestamps = []

    last_date = None
    # Diagnostics for trade quality
    total_positions = 0
    reached18_count = 0
    reached30_count = 0
    pos_reached18 = False
    pos_reached30 = False
    realized_at_entry: Optional[float] = None
    win_pnls: List[float] = []
    loss_pnls: List[float] = []
    daily_trades = 0

    # Debug counters for entry gating diagnostics
    stats = {
        "bars": 0,
        "trend_bars": 0,
        "mr_bars": 0,
        "long_signal": 0,
        "short_signal": 0,
        "skipped_atr": 0,
        "skipped_overext": 0,
        "skipped_pullback": 0,
        "skipped_no_breakout": 0,
        "skipped_mr_z": 0,
    }

    # Cooldowns per direction
    last_stop_i_long = -10**9
    last_stop_i_short = -10**9
    last_entry_i_long = -10**9
    last_entry_i_short = -10**9
    COOLDOWN_BARS = 3

    # Mode hysteresis
    last_mode: Optional[str] = None  # 'TREND', 'MR', or None

    def decide_trading_mode(row, prev_mode: Optional[str] = None) -> str:
        """TREND/MR classifier with ADX-gated votes and hysteresis.
        - MR votes: ADX low; and/or (HL very short while ADX low); and/or (BB compressed while ADX low)
        - TREND votes: ADX high; or HL long; or BB expanding while ADX moderate/high
        Decision:
          if TrendScore>=1 and MRScore==0 -> TREND
          elif MRScore>=2 and TrendScore==0 -> MR
          else -> prev_mode or TREND
        """
        hl = row.get("half_life")
        adx = row.get("adx_15m")
        bb_bw = row.get("bb_bw_5")
        bb_bw_ma = row.get("bb_bw_ma_5")

        try:
            hl = float(hl) if hl is not None else np.nan
        except Exception:
            hl = np.nan
        try:
            adx = float(adx) if adx is not None else np.nan
        except Exception:
            adx = np.nan
        try:
            bb_bw = float(bb_bw) if bb_bw is not None else np.nan
            bb_bw_ma = float(bb_bw_ma) if bb_bw_ma is not None else np.nan
        except Exception:
            bb_bw = np.nan; bb_bw_ma = np.nan

        TrendScore = 0
        MRScore = 0

        # ADX contribution
        if not np.isnan(adx):
            if adx >= ADX_TREND_MIN:
                TrendScore += 1
            elif adx <= ADX_MR_MAX:
                MRScore += 1

        # Half-life contribution (MR only when ADX low; TREND if long HL regardless)
        if not np.isnan(hl) and not np.isnan(adx):
            if adx <= ADX_MR_MAX and hl <= HL_MR_MAX:
                MRScore += 1
            elif hl >= HL_TREND_MIN:
                TrendScore += 1
        elif not np.isnan(hl) and hl >= HL_TREND_MIN:
            TrendScore += 1

        # Bollinger compression/expansion with ADX gating
        if (not np.isnan(bb_bw)) and (not np.isnan(bb_bw_ma)) and bb_bw_ma != 0:
            ratio = bb_bw / bb_bw_ma
            if ratio <= BB_COMPRESS_RATIO and (np.isnan(adx) or adx <= ADX_MR_MAX):
                MRScore += 1
            elif ratio >= BB_EXPAND_RATIO and (np.isnan(adx) or adx >= ADX_TREND_MIN):
                TrendScore += 1

        prev_mode = prev_mode or "TREND"
        if TrendScore >= 1 and MRScore == 0:
            return "TREND"
        if MRScore >= 2 and TrendScore == 0:
            return "MR"
        return prev_mode

    position_mode: Optional[str] = None

    for i in range(len(df)):
        row = df.iloc[i]
        ts = row.name  # timestamp
        date = ts.date()

        # reset daily PnL & flags on new calendar day
        if last_date is None or date != last_date:
            daily_pnl = 0.0
            trading_enabled = True
            last_date = date
            daily_trades = 0

        # compute regime
        regime = classify_regime(row)
        close = row["Close"]
        high = row["High"]
        low = row["Low"]
        open_ = row["Open"]

        # =====================
        # Manage open position
        # =====================
        if position != 0:
            qty = abs(position)
            direction = "LONG" if position > 0 else "SHORT"
            assert entry_price is not None
            assert stop_price is not None
            assert entry_index is not None

            exited = False
            exit_reason = ""
            exit_price = close  # default if time/regime/other exit

            # Update MFE (favorable excursion) since entry
            if mfe_price is None:
                mfe_price = entry_price
            if direction == "LONG":
                mfe_price = max(mfe_price, high)
            else:
                mfe_price = min(mfe_price, low)

            # 0) Breakeven and trailing using entry ATR (live-aligned)
            if not exited and atr_at_entry is not None and atr_at_entry > 0:
                move_fav = (close - entry_price) if direction == "LONG" else (entry_price - close)
                # Initial stop from entry ATR (kept from entry)
                if direction == "LONG":
                    base_stop = entry_price - ATR_STOP_MULT * atr_at_entry
                    if base_stop > stop_price + MIN_STOP_STEP_ATR * atr_at_entry:
                        stop_price = base_stop
                else:
                    base_stop = entry_price + ATR_STOP_MULT * atr_at_entry
                    if base_stop < stop_price - MIN_STOP_STEP_ATR * atr_at_entry:
                        stop_price = base_stop

                # Breakeven once price moves in favor by threshold (MR may use earlier trigger)
                BE_TRIG_LOCAL = (MR_BE_TRIGGER_ATR if position_mode == "MR" else BE_TRIGGER_ATR)
                BE_OFF_LOCAL = (MR_BE_OFFSET_ATR if position_mode == "MR" else BE_OFFSET_ATR)
                if move_fav >= BE_TRIG_LOCAL * atr_at_entry:
                    if direction == "LONG":
                        target = entry_price + BE_OFF_LOCAL * atr_at_entry
                        if target > stop_price + MIN_STOP_STEP_ATR * atr_at_entry:
                            stop_price = target
                    else:
                        target = entry_price - BE_OFF_LOCAL * atr_at_entry
                        if target < stop_price - MIN_STOP_STEP_ATR * atr_at_entry:
                            stop_price = target

                # Tiered ATR trailing using MFE
                if move_fav >= TRAIL1_TRIGGER_ATR * atr_at_entry:
                    target = (mfe_price - TRAIL1_MULT * atr_at_entry) if direction == "LONG" else (mfe_price + TRAIL1_MULT * atr_at_entry)
                    if direction == "LONG":
                        if target > stop_price + MIN_STOP_STEP_ATR * atr_at_entry:
                            stop_price = target
                    else:
                        if target < stop_price - MIN_STOP_STEP_ATR * atr_at_entry:
                            stop_price = target
                if move_fav >= TRAIL2_TRIGGER_ATR * atr_at_entry:
                    target = (mfe_price - TRAIL2_MULT * atr_at_entry) if direction == "LONG" else (mfe_price + TRAIL2_MULT * atr_at_entry)
                    if direction == "LONG":
                        if target > stop_price + MIN_STOP_STEP_ATR * atr_at_entry:
                            stop_price = target
                    else:
                        if target < stop_price - MIN_STOP_STEP_ATR * atr_at_entry:
                            stop_price = target
                if move_fav >= TRAIL3_TRIGGER_ATR * atr_at_entry:
                    target = (mfe_price - TRAIL3_MULT * atr_at_entry) if direction == "LONG" else (mfe_price + TRAIL3_MULT * atr_at_entry)
                    if direction == "LONG":
                        if target > stop_price + MIN_STOP_STEP_ATR * atr_at_entry:
                            stop_price = target
                    else:
                        if target < stop_price - MIN_STOP_STEP_ATR * atr_at_entry:
                            stop_price = target

                # Ratchet stop once move_fav is large enough
                if move_fav >= 3.0 * atr_at_entry:
                    if direction == "LONG":
                        target = entry_price + 0.5 * atr_at_entry
                        if target > stop_price + MIN_STOP_STEP_ATR * atr_at_entry:
                            stop_price = target
                    else:
                        target = entry_price - 0.5 * atr_at_entry
                        if target < stop_price - MIN_STOP_STEP_ATR * atr_at_entry:
                            stop_price = target

                # Winner-protection: if MFE reaches SAFE_MFE_POINTS (points),
                # enforce a minimum stop at breakeven + small offset so a big winner
                # cannot become a full loser.
                if SAFE_MFE_POINTS and entry_price is not None and mfe_price is not None:
                    if direction == "LONG" and (mfe_price - entry_price) >= SAFE_MFE_POINTS:
                        min_stop = entry_price + SAFE_BE_OFFSET_POINTS
                        if (stop_price is None) or (min_stop > stop_price):
                            stop_price = min_stop
                    elif direction == "SHORT" and (entry_price - mfe_price) >= SAFE_MFE_POINTS:
                        min_stop = entry_price - SAFE_BE_OFFSET_POINTS
                        if (stop_price is None) or (min_stop < stop_price):
                            stop_price = min_stop

            # Pyramiding: add on favorable moves at ATR steps
            if PYRAMID_ENABLED and atr_at_entry and not np.isnan(atr_at_entry) and qty < MAX_CONTRACTS:
                try:
                    pyr_count
                except NameError:
                    pyr_count = 0
                next_level = (pyr_count + 1) * PYRAMID_STEP_ATR * atr_at_entry
                if move_fav >= next_level:
                    add_qty = min(PYRAMID_ADD_QTY, MAX_CONTRACTS - qty)
                    if add_qty > 0:
                        qty += add_qty
                        position = qty if direction == "LONG" else -qty
                        pyr_count += 1
                        max_qty_local = max(max_qty_local, qty)

            # 0b) Hard dollar stop (additional tail guard)
            HARD_DLR_STOP = 150.0
            if not exited:
                unreal_now = ((close - entry_price) if direction == "LONG" else (entry_price - close)) * qty * MULTIPLIER
                if unreal_now <= -HARD_DLR_STOP:
                    exit_reason = "HARD_DLR_STOP"
                    exit_price = close
                    exited = True

            # 1) Hard stop
            if not exited:
                if direction == "LONG":
                    if low <= stop_price:
                        exit_price = stop_price
                        exit_reason = "STOP"
                        exited = True
                else:  # SHORT
                    if high >= stop_price:
                        exit_price = stop_price
                        exit_reason = "STOP"
                        exited = True

            # 2) Stop breach at bar extremes (intra-bar check)
            if not exited and stop_price is not None:
                if direction == "LONG" and low <= stop_price:
                    exit_price = stop_price
                    exit_reason = "STOP"
                    exited = True
                elif direction == "SHORT" and high >= stop_price:
                    exit_price = stop_price
                    exit_reason = "STOP"
                    exited = True

            # TREND partials at +1.5×ATR and +3.0×ATR
            if not exited and position_mode == "TREND" and qty >= 2 and atr_at_entry and not np.isnan(atr_at_entry):
                move_fav = (close - entry_price) if direction == "LONG" else (entry_price - close)
                # Initialize flags on first use
                if 'pos_trend_partial1_done' not in locals():
                    pos_trend_partial1_done = False
                if 'pos_trend_partial2_done' not in locals():
                    pos_trend_partial2_done = False

                # First partial at 1.5×ATR
                if (not pos_trend_partial1_done) and move_fav >= 1.5 * atr_at_entry:
                    part_qty = max(1, qty // 3)
                    px = close
                    pnl_part = ((px - entry_price) if direction == "LONG" else (entry_price - px)) * part_qty * MULTIPLIER
                    realized_pnl += pnl_part
                    daily_pnl += pnl_part
                    trades.append(Trade(
                        direction=direction,
                        entry_time=df.index[entry_index],
                        exit_time=ts,
                        entry_price=entry_price,
                        exit_price=px,
                        qty=part_qty,
                        pnl=pnl_part,
                        reason="TREND_PARTIAL_ATR",
                        initial_stop=(entry_price - stop_points_entry) if direction == "LONG" else (entry_price + stop_points_entry),
                        atr_at_entry=atr_at_entry,
                        stop_points_entry=stop_points_entry,
                        mode=position_mode,
                        strategy=position_strategy if 'position_strategy' in locals() else position_mode,
                        entry_reason=position_entry_reason if 'position_entry_reason' in locals() else None,
                        is_full_exit=False,
                        mfe_points=(max(0.0, float(mfe_price - entry_price)) if entry_price is not None and mfe_price is not None and direction == "LONG" else (max(0.0, float(entry_price - mfe_price)) if entry_price is not None and mfe_price is not None else None)),
                        mfe_price_at_exit=float(mfe_price) if mfe_price is not None else None,
                    ))
                    qty = qty - part_qty
                    position = qty if direction == "LONG" else -qty
                    pos_trend_partial1_done = True

                # Second partial at 3.0×ATR
                if (not exited) and qty >= 2 and (not pos_trend_partial2_done) and move_fav >= 3.0 * atr_at_entry:
                    part_qty = max(1, qty // 3)
                    px = close
                    pnl_part = ((px - entry_price) if direction == "LONG" else (entry_price - px)) * part_qty * MULTIPLIER
                    realized_pnl += pnl_part
                    daily_pnl += pnl_part
                    trades.append(Trade(
                        direction=direction,
                        entry_time=df.index[entry_index],
                        exit_time=ts,
                        entry_price=entry_price,
                        exit_price=px,
                        qty=part_qty,
                        pnl=pnl_part,
                        reason="TREND_PARTIAL2_ATR",
                        initial_stop=(entry_price - stop_points_entry) if direction == "LONG" else (entry_price + stop_points_entry),
                        atr_at_entry=atr_at_entry,
                        stop_points_entry=stop_points_entry,
                        mode=position_mode,
                        strategy=position_strategy if 'position_strategy' in locals() else position_mode,
                        entry_reason=position_entry_reason if 'position_entry_reason' in locals() else None,
                        is_full_exit=False,
                        mfe_points=(max(0.0, float(mfe_price - entry_price)) if entry_price is not None and mfe_price is not None and direction == "LONG" else (max(0.0, float(entry_price - mfe_price)) if entry_price is not None and mfe_price is not None else None)),
                        mfe_price_at_exit=float(mfe_price) if mfe_price is not None else None,
                    ))
                    qty = qty - part_qty
                    position = qty if direction == "LONG" else -qty
                    pos_trend_partial2_done = True

            # MR exits: exit on Z reversion toward zero; tighter ATR stop applied at entry
            if not exited and position_mode == "MR":
                z_here = get_mr_z(row)
                if z_here is not None and not np.isnan(z_here):
                    # Close near zero: |z| <= MR_Z_TP_ZONE
                    if direction == "LONG" and z_here >= -MR_Z_TP_ZONE:
                        exit_reason = "MR_Z_REVERSION"
                        exit_price = close
                        exited = True
                    elif direction == "SHORT" and z_here <= MR_Z_TP_ZONE:
                        exit_reason = "MR_Z_REVERSION"
                        exit_price = close
                        exited = True

            # 3) Exhaustion: prefer runner (partial) and require ADX rollover + profit
            EXH_MIN_PROFIT_ATR = 0.5
            if not exited:
                ema_1h = row.get("ema_1h_50", np.nan)
                atr_now = row.get("atr", np.nan)
                if (
                    ema_1h is not None and not np.isnan(ema_1h)
                    and atr_now is not None and not np.isnan(atr_now)
                    and atr_now > 0
                ):
                    unreal_now = ((close - entry_price) if direction == "LONG" else (entry_price - close)) * qty * MULTIPLIER
                    min_profit_dlr = EXH_MIN_PROFIT_ATR * atr_now * MULTIPLIER * qty
                    adx_now = row.get("adx14", np.nan)
                    adx_prev = df["adx14"].iloc[i-1] if i > 0 else np.nan
                    adx_rollover = (not np.isnan(adx_now)) and (not np.isnan(adx_prev)) and (adx_now < adx_prev)

                    trigger_long = direction == "LONG" and close > ema_1h + EXHAUSTION_ATR_MULT * atr_now
                    trigger_short = direction == "SHORT" and close < ema_1h - EXHAUSTION_ATR_MULT * atr_now
                    if (trigger_long or trigger_short) and unreal_now >= min_profit_dlr and adx_rollover and qty >= 2:
                        part_qty = max(1, qty // 2)
                        px = close
                        pnl_part = ((px - entry_price) if direction == "LONG" else (entry_price - px)) * part_qty * MULTIPLIER
                        realized_pnl += pnl_part
                        daily_pnl += pnl_part
                        init_stop = (
                            (entry_price - stop_points_entry)
                            if (stop_points_entry is not None and entry_price is not None and direction == "LONG")
                            else (
                                (entry_price + stop_points_entry)
                                if (stop_points_entry is not None and entry_price is not None and direction == "SHORT")
                                else None
                            )
                        )
                        trades.append(Trade(
                            direction=direction,
                            entry_time=df.index[entry_index],
                            exit_time=ts,
                            entry_price=entry_price,
                            exit_price=px,
                            qty=part_qty,
                            pnl=pnl_part,
                            reason="EXHAUSTION_PARTIAL",
                            initial_stop=init_stop,
                            atr_at_entry=atr_at_entry,
                            stop_points_entry=stop_points_entry,
                            mode=position_mode,
                            is_full_exit=False,
                            mfe_points=(max(0.0, float(mfe_price - entry_price)) if entry_price is not None and mfe_price is not None and direction == "LONG" else (max(0.0, float(entry_price - mfe_price)) if entry_price is not None and mfe_price is not None else None)),
                            mfe_price_at_exit=float(mfe_price) if mfe_price is not None else None,
                        ))
                        # Reduce position; keep runner
                        position = (qty - part_qty) if direction == "LONG" else -(qty - part_qty)
                        qty = abs(position)

            # 4) force-flat time (maintenance window)
            if not exited and past_force_flat(ts):
                exit_reason = "FORCE_CLOSE_EOD"
                exit_price = close
                exited = True

            # 5) time-stop (adaptive)
            if not exited and entry_time is not None:
                minutes_in_trade = (ts - entry_time).total_seconds() / 60.0
                eff_time_stop = TIME_STOP_MIN
                try:
                    move_fav_ts = (close - entry_price) if direction == "LONG" else (entry_price - close)
                    if atr_at_entry and move_fav_ts >= 1.0 * atr_at_entry:
                        eff_time_stop = TIME_STOP_MIN + TIME_STOP_EXT_MIN
                except Exception:
                    pass
                if minutes_in_trade >= eff_time_stop:
                    exit_reason = "TIME_STOP"
                    exit_price = close
                    exited = True

            if exited:
                if direction == "LONG":
                    pnl = (exit_price - entry_price) * qty * MULTIPLIER
                else:
                    pnl = (entry_price - exit_price) * qty * MULTIPLIER

                realized_pnl += pnl
                daily_pnl += pnl

                # Derive initial stop from entry context for logging
                init_stop: Optional[float] = None
                if entry_price is not None and stop_points_entry is not None:
                    init_stop = entry_price - stop_points_entry if direction == "LONG" else entry_price + stop_points_entry

                # MFE in points
                mfe_pts: Optional[float] = None
                if mfe_price is not None and entry_price is not None:
                    if direction == "LONG":
                        mfe_pts = max(0.0, float(mfe_price) - float(entry_price))
                    else:
                        mfe_pts = max(0.0, float(entry_price) - float(mfe_price))

                trades.append(
                    Trade(
                        direction=direction,
                        entry_time=df.index[entry_index],
                        exit_time=ts,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        qty=qty,
                        pnl=pnl,
                        reason=exit_reason,
                        initial_stop=init_stop,
                        atr_at_entry=atr_at_entry,
                        stop_points_entry=stop_points_entry,
                        mode=position_mode,
                        is_full_exit=True,
                        mfe_points=mfe_pts,
                        mfe_price_at_exit=float(mfe_price) if mfe_price is not None else None,
                        entry_qty=entry_qty_local,
                        max_qty=max_qty_local,
                    )
                )

                # flat position
                position = 0
                entry_price = stop_price = target_price = None
                entry_index = None
                atr_at_entry = None
                entry_time = None
                mfe_price = None
                stop_points_entry = None

                # record cooldown for same-direction re-entry after stop
                if exit_reason == "STOP":
                    if direction == "LONG":
                        last_stop_i_long = i
                    else:
                        last_stop_i_short = i

                # finalize diagnostics for this closed position
                if realized_at_entry is not None:
                    pos_pnl = realized_pnl - realized_at_entry
                    if pos_reached18:
                        reached18_count += 1
                    if pos_reached30:
                        reached30_count += 1
                    if pos_pnl > 0:
                        win_pnls.append(pos_pnl)
                    else:
                        loss_pnls.append(pos_pnl)
                    pos_reached18 = False
                    pos_reached30 = False
                    realized_at_entry = None

                # After any full exit, enforce daily loss limit
                if daily_pnl <= DAILY_LOSS_LIMIT:
                    trading_enabled = False

            # Partial profit logic (optional)
            elif TAKE_PARTIAL and position != 0 and atr_at_entry and not np.isnan(atr_at_entry) and abs(position) >= 2:
                direction = "LONG" if position > 0 else "SHORT"
                part_qty = max(1, abs(position) // 2)
                if direction == "LONG" and high >= entry_price + PARTIAL_AT_POINTS:
                    px = entry_price + PARTIAL_AT_POINTS
                    pnl_part = (px - entry_price) * part_qty * MULTIPLIER
                    realized_pnl += pnl_part
                    daily_pnl += pnl_part
                    # Partial realization record (keep initial stop context)
                    init_stop = (
                        (entry_price - stop_points_entry)
                        if (stop_points_entry is not None and entry_price is not None)
                        else (entry_price - 2.0 * atr_at_entry if atr_at_entry is not None and entry_price is not None else None)
                    )
                    trades.append(Trade(
                        direction=direction,
                        entry_time=df.index[entry_index],
                        exit_time=ts,
                        entry_price=entry_price,
                        exit_price=px,
                        qty=part_qty,
                        pnl=pnl_part,
                        reason="PARTIAL_TP",
                        initial_stop=init_stop,
                        atr_at_entry=atr_at_entry,
                        stop_points_entry=stop_points_entry,
                        mode=position_mode,
                        strategy=position_strategy if 'position_strategy' in locals() else position_mode,
                        entry_reason=position_entry_reason if 'position_entry_reason' in locals() else None,
                        is_full_exit=False,
                        mfe_points=(max(0.0, float(mfe_price - entry_price)) if entry_price is not None and mfe_price is not None and direction == "LONG" else (max(0.0, float(entry_price - mfe_price)) if entry_price is not None and mfe_price is not None else None)),
                        mfe_price_at_exit=float(mfe_price) if mfe_price is not None else None,
                    ))
                    position -= part_qty
                elif direction == "SHORT" and low <= entry_price - PARTIAL_AT_POINTS:
                    px = entry_price - PARTIAL_AT_POINTS
                    pnl_part = (entry_price - px) * part_qty * MULTIPLIER
                    realized_pnl += pnl_part
                    daily_pnl += pnl_part
                    init_stop = (
                        (entry_price + stop_points_entry)
                        if (stop_points_entry is not None and entry_price is not None)
                        else (entry_price + 2.0 * atr_at_entry if atr_at_entry is not None and entry_price is not None else None)
                    )
                    trades.append(Trade(
                        direction=direction,
                        entry_time=df.index[entry_index],
                        exit_time=ts,
                        entry_price=entry_price,
                        exit_price=px,
                        qty=part_qty,
                        pnl=pnl_part,
                        reason="PARTIAL_TP",
                        initial_stop=init_stop,
                        atr_at_entry=atr_at_entry,
                        stop_points_entry=stop_points_entry,
                        mode=position_mode,
                        strategy=position_strategy if 'position_strategy' in locals() else position_mode,
                        entry_reason=position_entry_reason if 'position_entry_reason' in locals() else None,
                        is_full_exit=False,
                        mfe_points=(max(0.0, float(mfe_price - entry_price)) if entry_price is not None and mfe_price is not None and direction == "LONG" else (max(0.0, float(entry_price - mfe_price)) if entry_price is not None and mfe_price is not None else None)),
                        mfe_price_at_exit=float(mfe_price) if mfe_price is not None else None,
                    ))
                    position += part_qty

                if daily_pnl <= DAILY_LOSS_LIMIT:
                    trading_enabled = False

        # Update equity and enforce max drawdown liquidation
        current_equity = STARTING_CAPITAL + realized_pnl
        if current_equity > peak_equity:
            peak_equity = current_equity
        if peak_equity - current_equity >= MAX_DD_LIMIT:
            # Force flat and stop further trading
            if position != 0:
                qty = abs(position)
                direction = "LONG" if position > 0 else "SHORT"
                exit_price = close
                if direction == "LONG":
                    pnl = (exit_price - entry_price) * qty * MULTIPLIER
                else:
                    pnl = (entry_price - exit_price) * qty * MULTIPLIER
                realized_pnl += pnl
                daily_pnl += pnl
                init_stop = None
                if entry_price is not None and atr_at_entry is not None:
                    init_stop = entry_price - 2.0 * atr_at_entry if direction == "LONG" else entry_price + 2.0 * atr_at_entry
                trades.append(
                    Trade(
                        direction=direction,
                        entry_time=df.index[entry_index] if entry_index is not None else ts,
                        exit_time=ts,
                        entry_price=entry_price if entry_price is not None else close,
                        exit_price=exit_price,
                        qty=qty,
                        pnl=pnl,
                        reason="MAX_DD_LIQUIDATION",
                        mode=position_mode,
                        strategy=position_strategy if 'position_strategy' in locals() else position_mode,
                        entry_reason=position_entry_reason if 'position_entry_reason' in locals() else None,
                        initial_stop=init_stop,
                        atr_at_entry=atr_at_entry,
                        stop_points_entry=stop_points_entry,
                    )
                )
                position = 0
                entry_price = stop_price = target_price = None
                entry_index = None
                atr_at_entry = None
            trading_enabled = False

        # =====================
        # Consider new entries
        # =====================
        if (
            position == 0
            and trading_enabled
            and in_entry_session(ts)
        ):
            stats["bars"] += 1

            atr = row.get("atr", np.nan)
            donch_high = row.get("donch_high", np.nan)
            donch_low = row.get("donch_low", np.nan)
            ema_1h = row.get("ema_1h_50", np.nan)

            if atr is None or np.isnan(atr) or atr <= 0:
                continue
            if np.isnan(ema_1h):
                continue

            prev_mode_here = last_mode or "TREND"
            mode = decide_trading_mode(row, prev_mode=prev_mode_here)
            if mode != prev_mode_here:
                try:
                    ratio = float(row.get("bb_bw_5") or np.nan) / float(row.get("bb_bw_ma_5") or np.nan)
                except Exception:
                    ratio = np.nan
                try:
                    print(f"[{base_symbol}] mode change: {prev_mode_here} -> {mode}, half_life={float(row.get('half_life') or np.nan):.1f}, adx_15m={float(row.get('adx_15m') or np.nan):.1f}, bb_ratio={(ratio if not np.isnan(ratio) else float('nan')):.2f}")
                except Exception:
                    pass
            last_mode = mode
            # optional: attach for analysis
            try:
                df.loc[ts, 'mode'] = mode  # type: ignore[index]
            except Exception:
                pass
            # regime counts
            if mode == "TREND":
                stats["trend_bars"] += 1
            elif mode == "MR":
                stats["mr_bars"] += 1
            # ATR band: TREND uses per-symbol/global; MR for MNQ uses MR ATR band
            sym = (base_symbol or BASE_SYMBOL or "MNQ").upper()
            if mode == "MR" and MR_ENABLE and is_mnq_symbol(sym):
                if not atr_in_mr_band(float(atr)):
                    stats["skipped_atr"] += 1
                    continue
            else:
                if not atr_in_band_for_symbol(sym, float(atr)):
                    stats["skipped_atr"] += 1
                    continue

            long_signal = False
            short_signal = False

            if mode == "TREND":
                # Overextension advisory only (no hard block)
                # If you want to hard-block extreme entries again, re-enable the continue.
                # if is_overextended(row):
                #     stats["skipped_overext"] += 1
                #     continue
                # Early trend breakout in direction of hourly trend
                tsz = tick_size_for(sym)
                if regime == "UP" and (not np.isnan(donch_high)) and close > (donch_high + tsz):
                    if micro_pullback_ok(df, ts, "LONG"):
                        long_signal = True
                        entry_reason_tag = "BREAKOUT"
                        entry_strategy_tag = "TREND"
                    else:
                        stats["skipped_pullback"] += 1
                elif regime == "DOWN" and (not np.isnan(donch_low)) and close < (donch_low - tsz):
                    if micro_pullback_ok(df, ts, "SHORT"):
                        short_signal = True
                        entry_reason_tag = "BREAKOUT"
                        entry_strategy_tag = "TREND"
                    else:
                        stats["skipped_pullback"] += 1
                else:
                    stats["skipped_no_breakout"] += 1
            elif mode == "MR" and MR_ENABLE and is_mnq_symbol(sym):
                # Mean-reversion Z-score entries around EMA
                z_val = get_mr_z(row)
                if z_val is not None and not np.isnan(z_val):
                    if abs(z_val) >= MR_Z_ENTRY and abs(z_val) <= MR_Z_MAX_EXTREME:
                        if z_val <= -MR_Z_ENTRY:
                            long_signal = True
                            entry_reason_tag = "MR_Z_REVERSION"
                            entry_strategy_tag = "MR"
                        elif z_val >= MR_Z_ENTRY:
                            short_signal = True
                            entry_reason_tag = "MR_Z_REVERSION"
                            entry_strategy_tag = "MR"
                    else:
                        stats["skipped_mr_z"] += 1
                else:
                    stats["skipped_mr_z"] += 1

            if long_signal:
                stats["long_signal"] += 1
            if short_signal:
                stats["short_signal"] += 1

            # Directional cooldown per entry
            if long_signal and (i - last_entry_i_long) < COOLDOWN_BARS:
                stats["skipped_cooldown"] = stats.get("skipped_cooldown", 0) + 1
                long_signal = False
            if short_signal and (i - last_entry_i_short) < COOLDOWN_BARS:
                stats["skipped_cooldown"] = stats.get("skipped_cooldown", 0) + 1
                short_signal = False

            if long_signal or short_signal:
                direction = "LONG" if long_signal else "SHORT"

                # Directional cooldown after recent stop
                if direction == "LONG" and (i - last_stop_i_long) < COOLDOWN_BARS:
                    continue
                if direction == "SHORT" and (i - last_stop_i_short) < COOLDOWN_BARS:
                    continue

                # Position sizing (risk-based only)
                # ATR-based stop distance: tighter for MR and for TREND reduce to 1.5x ATR
                if mode == "MR" and MR_ENABLE and is_mnq_symbol(sym):
                    stop_points = max(0.25, MR_ATR_STOP_MULT * atr)
                else:
                    stop_points = 1.0 * atr
                if stop_points <= 0:
                    continue

                risk_per_contract = stop_points * MULTIPLIER
                if risk_per_contract <= 0:
                    continue

                qty = int(RISK_PER_TRADE / risk_per_contract)
                # Enforce minimum size even if risk-based qty is small
                qty = max(qty, MIN_QTY)
                qty = min(qty, MAX_CONTRACTS)

                if direction == "LONG":
                    stop_raw = close - stop_points
                    target_raw = close + 9999.0  # trail/logic will manage exit
                else:
                    stop_raw = close + stop_points
                    target_raw = close - 9999.0

                # open position
                position = qty if direction == "LONG" else -qty
                entry_price = close
                stop_price = stop_raw
                target_price = target_raw
                entry_index = i
                entry_time = ts
                atr_at_entry = atr
                stop_points_entry = stop_points
                mfe_price = entry_price
                total_positions += 1
                position_mode = mode
                try:
                    position_strategy = entry_strategy_tag if ('entry_strategy_tag' in locals()) else ("TREND" if position_mode == "TREND" else "MR")
                    position_entry_reason = entry_reason_tag if ('entry_reason_tag' in locals()) else None
                except Exception:
                    position_strategy = ("TREND" if position_mode == "TREND" else "MR")
                    position_entry_reason = None
                realized_at_entry = realized_pnl
                pos_reached18 = False
                pos_reached30 = False
                daily_trades += 1
                # Track entry and max qty for analytics
                entry_qty_local = qty
                max_qty_local = qty
                pyr_count = 0
                # update entry cooldown indices
                if direction == "LONG":
                    last_entry_i_long = i
                else:
                    last_entry_i_short = i

        # record equity
        equity_curve.append(STARTING_CAPITAL + realized_pnl)
        timestamps.append(ts)

    # Print diagnostics including skip reasons
    print("Debug counts:")
    for k in [
        "bars",
        "trend_bars",
        "mr_bars",
        "long_signal",
        "short_signal",
        "skipped_atr",
        "skipped_overext",
        "skipped_pullback",
        "skipped_no_breakout",
        "skipped_mr_z",
    ]:
        print(f"  {k}: {stats.get(k, 0)}")

    # Print diagnostics for % of positions reaching trailing tiers and avg win/loss
    if total_positions > 0:
        p18 = 100.0 * reached18_count / total_positions
        p30 = 100.0 * reached30_count / total_positions
        avg_win = float(np.mean(win_pnls)) if win_pnls else 0.0
        avg_loss = float(np.mean(loss_pnls)) if loss_pnls else 0.0
        print(f"Positions: {total_positions} | >=18pt: {p18:.1f}% | >=30pt: {p30:.1f}%")
        print(f"Avg win: {avg_win:.2f} | Avg loss: {avg_loss:.2f}")

    equity_series = pd.Series(equity_curve, index=timestamps)
    return trades, equity_series


def summarize_trades(trades: List[Trade]):
    if not trades:
        print("No trades taken.")
        return

    df_trades = pd.DataFrame([t.__dict__ for t in trades])
    total_trades = len(df_trades)
    wins = (df_trades["pnl"] > 0).sum()
    win_rate = wins / total_trades * 100.0
    avg_pnl = df_trades["pnl"].mean()
    med_pnl = df_trades["pnl"].median()
    best = df_trades["pnl"].max()
    worst = df_trades["pnl"].min()
    gross_profit = df_trades[df_trades["pnl"] > 0]["pnl"].sum()
    gross_loss = df_trades[df_trades["pnl"] < 0]["pnl"].sum()
    net = df_trades["pnl"].sum()
    # Simple performance metrics
    pnl_cum = df_trades["pnl"].cumsum()
    max_dd = (pnl_cum.cummax() - pnl_cum).max()
    print(f"Max drawdown: {max_dd:.2f}")

    print("===== Trade Summary =====")
    print(f"Total trades: {total_trades}")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Average PnL: {avg_pnl:.2f}")
    print(f"Median PnL: {med_pnl:.2f}")
    print(f"Best trade: {best:.2f}")
    print(f"Worst trade: {worst:.2f}")
    print(f"Gross profit: {gross_profit:.2f}")
    print(f"Gross loss: {gross_loss:.2f}")
    print(f"Net PnL: {net:.2f}")

    # Per-day trade counts and PnL (by entry date)
    try:
        dates = pd.to_datetime(df_trades["entry_time"]).dt.date
        df_trades["_entry_date"] = dates
        daily = (
            df_trades.groupby("_entry_date")["pnl"].agg(["count", "sum"]).rename(
                columns={"count": "trades", "sum": "pnl"}
            )
        )
        print("\n===== Trades per day =====")
        for d, row_day in daily.iterrows():
            print(f"{d}: {int(row_day['trades'])} trades, PnL={row_day['pnl']:.2f}")
    except Exception as e:
        print(f"Daily breakdown error: {e}")

    # Losing trades that were in profit (MFE>0)
    try:
        losers = df_trades[(df_trades["pnl"] < 0) & (df_trades["is_full_exit"] == True)]
        losers_were_winning = losers[losers["mfe_points"].fillna(0) > 0]
        n_losers = int(len(losers))
        n_were_win = int(len(losers_were_winning))
        pct = (100.0 * n_were_win / n_losers) if n_losers > 0 else 0.0
        print(f"\nLosing full exits that were winning at some point: {n_were_win}/{n_losers} ({pct:.1f}%)")
        if n_were_win > 0:
            print(
                f"MFE (pts) on those losers: median={losers_were_winning['mfe_points'].median():.2f}, mean={losers_were_winning['mfe_points'].mean():.2f}"
            )
    except Exception as e:
        print(f"Analysis error (MFE on losers): {e}")

    # Daily breakdown (full exits only)
    try:
        df_full = df_trades[df_trades.get("is_full_exit", True) == True].copy()
        if not df_full.empty and "exit_time" in df_full.columns:
            # normalize date
            et = pd.to_datetime(df_full["exit_time"], errors="coerce")
            try:
                et = et.dt.tz_convert("US/Central")
            except Exception:
                pass
            df_full["date"] = et.dt.date
            daily = df_full.groupby("date")["pnl"].agg(["count", "sum"]).reset_index()
            print("\nDaily breakdown (full exits):")
            for _, r in daily.iterrows():
                print(f"  {r['date']}: trades={int(r['count'])} pnl=${float(r['sum']):.2f}")
            # save for analysis
            try:
                daily.to_csv("daily_breakdown.csv", index=False)
            except Exception:
                pass
    except Exception as e:
        print(f"Daily breakdown error: {e}")

    print("\nSample trades:")
    print(df_trades.head())

    # Export trades for inspection with a safe fallback name if locked
    try:
        df_trades.to_csv("mnq_trades.csv", index=False)
        print("Saved trades to mnq_trades.csv")
    except PermissionError:
        alt_name = f"mnq_trades_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            df_trades.to_csv(alt_name, index=False)
            print(f"Saved trades to {alt_name} (mnq_trades.csv was locked).")
        except Exception as e:
            print(f"Could not save trades CSV: {e}")


def trades_to_df(trades: List[Trade]) -> pd.DataFrame:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        raise
    return pd.DataFrame([t.__dict__ for t in trades])


def _last_hours_window(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = df.index.max() - pd.Timedelta(hours=int(hours))
    return df[df.index >= cutoff]


def _collect_entry_events(trades: List[Trade], start_ts: pd.Timestamp, end_ts: pd.Timestamp):
    # Deduplicate entries by (entry_time, direction, entry_price)
    seen = set()
    events = []
    for t in trades:
        et = pd.Timestamp(t.entry_time)
        if et >= start_ts and et <= end_ts:
            key = (et.value, t.direction, float(t.entry_price))
            if key not in seen:
                seen.add(key)
                events.append({
                    "time": et,
                    "price": float(t.entry_price),
                    "direction": t.direction,
                })
    # sort by time
    events.sort(key=lambda x: x["time"])
    return events


def plot_price_with_executions(df: pd.DataFrame, trades: List[Trade], hours: int = 24, symbol: str = "MNQ"):
    if df.empty:
        print("No price data to plot.")
        return
    win = _last_hours_window(df, hours)
    if win.empty:
        print(f"No data within last {hours} hours to plot.")
        return

    start_ts = win.index.min()
    end_ts = win.index.max()
    # Build entry events and collect exits (partials and full)
    entries = _collect_entry_events(trades, start_ts, end_ts)
    exits = []
    for t in trades:
        xt = pd.Timestamp(t.exit_time)
        if xt >= start_ts and xt <= end_ts:
            exits.append({
                "time": xt,
                "price": float(t.exit_price),
                "direction": t.direction,
                "is_full": bool(t.is_full_exit),
                "reason": t.reason,
                "qty": int(t.qty) if t.qty is not None else None,
                "pnl": float(getattr(t, "pnl", np.nan)) if getattr(t, "pnl", None) is not None else np.nan,
            })

    # Compute 24h PnL and counts
    try:
        pnl_24 = float(sum([getattr(t, "pnl", 0.0) for t in trades if pd.Timestamp(t.exit_time) >= start_ts and pd.Timestamp(t.exit_time) <= end_ts]))
        n_full = int(sum([1 for t in trades if pd.Timestamp(t.exit_time) >= start_ts and pd.Timestamp(t.exit_time) <= end_ts and bool(t.is_full_exit)]))
        n_part = int(sum([1 for t in trades if pd.Timestamp(t.exit_time) >= start_ts and pd.Timestamp(t.exit_time) <= end_ts and not bool(t.is_full_exit)]))
    except Exception:
        pnl_24, n_full, n_part = 0.0, 0, 0

    # Estimate entry sizes per entry_time using max of entry_qty or qty
    entry_size_map = {}
    for t in trades:
        et = pd.Timestamp(t.entry_time)
        if et < start_ts or et > end_ts:
            continue
        size_hint = None
        if getattr(t, "entry_qty", None) is not None and not pd.isna(t.entry_qty):
            try:
                size_hint = int(t.entry_qty)
            except Exception:
                pass
        if size_hint is None:
            try:
                size_hint = max(int(getattr(t, "max_qty", 0) or 0), int(getattr(t, "qty", 0) or 0))
            except Exception:
                size_hint = int(getattr(t, "qty", 0) or 0)
        prev = entry_size_map.get(et)
        if prev is None or (isinstance(size_hint, int) and size_hint > prev):
            entry_size_map[et] = size_hint

    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(14, 6))

    # Candlestick rendering
    x = mdates.date2num(win.index.to_pydatetime())
    o = win["Open"].values
    h = win["High"].values
    l = win["Low"].values
    c = win["Close"].values
    if len(x) > 1:
        width = (x[1] - x[0]) * 0.6
    else:
        width = 0.0005
    for xi, oi, hi, li, ci in zip(x, o, h, l, c):
        up = ci >= oi
        color = "#2ca02c" if up else "#d62728"
        ax.vlines(xi, li, hi, color=color, linewidth=0.6, alpha=0.9)
        body_y = min(oi, ci)
        body_h = abs(ci - oi)
        if body_h < 1e-6:
            ax.hlines(ci, xi - width/2, xi + width/2, color=color, linewidth=1.2, alpha=0.9)
        else:
            rect = Rectangle((xi - width/2, body_y), width, body_h, facecolor=color, edgecolor=color, alpha=0.8)
            ax.add_patch(rect)
    ax.set_xlim(win.index[0], win.index[-1])
    ax.xaxis_date()

    # Entries
    for ev in entries:
        if ev["direction"] == "LONG":
            ax.scatter(ev["time"], ev["price"], marker="^", s=64, color="#2ca02c", label=None, zorder=5)
        else:
            ax.scatter(ev["time"], ev["price"], marker="v", s=64, color="#d62728", label=None, zorder=5)
        # Annotate with direction and size hint
        size_hint = entry_size_map.get(pd.Timestamp(ev["time"]))
        if size_hint and size_hint > 0:
            label = ("L" if ev["direction"] == "LONG" else "S") + f" x{size_hint}"
            ax.annotate(label, xy=(ev["time"], ev["price"]), xytext=(0, -12 if ev["direction"] == "SHORT" else 12), textcoords='offset points', color="#333333", fontsize=7, ha='center', va='bottom' if ev["direction"] == "LONG" else 'top', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

    # Exits and partials (annotate profitability)
    for ev in exits:
        pnl = ev.get("pnl")
        pnl_color = "#2ca02c" if (pnl is not None and not pd.isna(pnl) and pnl > 0) else ("#d62728" if (pnl is not None and not pd.isna(pnl) and pnl < 0) else "#7f7f7f")
        if not ev["is_full"]:
            ax.scatter(ev["time"], ev["price"], marker="o", s=38, facecolors="none", edgecolors=pnl_color, label=None, zorder=6)
            # annotate partial qty and pnl
            try:
                parts = []
                qty = ev.get("qty")
                if qty:
                    parts.append(f"x{qty}")
                if pnl is not None and not pd.isna(pnl):
                    parts.append(("+$" if pnl >= 0 else "-$") + f"{abs(pnl):.0f}")
                if parts:
                    ax.annotate("part " + " ".join(parts), xy=(ev["time"], ev["price"]), xytext=(6, 6), textcoords='offset points', fontsize=7, color=pnl_color, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6))
            except Exception:
                pass
        else:
            ax.scatter(ev["time"], ev["price"], marker="x", s=52, color=pnl_color, label=None, zorder=6)
            # annotate reason, qty, and pnl
            try:
                parts = []
                reason = ev.get("reason")
                if reason:
                    parts.append(str(reason))
                qty = ev.get("qty")
                if qty:
                    parts.append(f"x{qty}")
                if pnl is not None and not pd.isna(pnl):
                    parts.append(("+$" if pnl >= 0 else "-$") + f"{abs(pnl):.0f}")
                if parts:
                    ax.annotate(" ".join(parts), xy=(ev["time"], ev["price"]), xytext=(6, -10), textcoords='offset points', fontsize=7, color=pnl_color, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6))
            except Exception:
                pass

    ax.set_title(f"{symbol} 1m Candles with Executions (Last {hours}h)")
    ax.set_xlabel("Time (US/Central)")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    # Left-top info box with last-N-hours PnL
    try:
        info = f"PnL {hours}h: ${pnl_24:.2f}\nFull: {n_full}  Partials: {n_part}"
        ax.text(0.01, 0.99, info, transform=ax.transAxes, va='top', ha='left', fontsize=8, bbox=dict(boxstyle='round', fc='white', ec='#999999', alpha=0.8))
    except Exception:
        pass
    # Build a simple legend proxy
    import matplotlib.lines as mlines
    long_entry = mlines.Line2D([], [], color="#2ca02c", marker='^', linestyle='None', markersize=8, label='Entry LONG')
    short_entry = mlines.Line2D([], [], color="#d62728", marker='v', linestyle='None', markersize=8, label='Entry SHORT')
    partial = mlines.Line2D([], [], color="#ff7f0e", marker='o', fillstyle='none', linestyle='None', markersize=6, label='Partial Exit')
    full_stop = mlines.Line2D([], [], color="#d62728", marker='x', linestyle='None', markersize=7, label='Full Exit (STOP)')
    full_other = mlines.Line2D([], [], color="#1f77b4", marker='x', linestyle='None', markersize=7, label='Full Exit (Other)')
    ax.legend(handles=[long_entry, short_entry, partial, full_stop, full_other], loc='best')
    plt.tight_layout()
    if os.environ.get("BACKTEST_SAVE_PLOTS", "").strip():
        fname = f"{symbol.lower()}_executions_last{hours}h.png"
        try:
            plt.savefig(fname, dpi=150)
            print(f"Saved execution plot to {fname}")
        except Exception as e:
            print(f"Could not save execution plot: {e}")
        finally:
            plt.close(fig)
    else:
        plt.show()


def plot_equity(equity: pd.Series):
    plt.figure(figsize=(10, 5))
    plt.plot(equity.index, equity.values)
    plt.title("Equity Curve (MNQ Trend-Following Strategy)")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.tight_layout()
    if os.environ.get("BACKTEST_SAVE_PLOTS", "").strip():
        try:
            plt.savefig("equity_curve.png", dpi=150)
            print("Saved equity curve to equity_curve.png")
        except Exception as e:
            print(f"Could not save equity curve: {e}")
        finally:
            plt.close()
    else:
        plt.show()


def plot_daily_pnl(equity: pd.Series, target_per_day: float = 1000.0):
    # Resample to daily end-of-day equity and compute day-over-day change
    eq_daily = equity.resample('1D').last().dropna()
    daily_chg = eq_daily.diff().dropna()

    plt.figure(figsize=(10, 4))
    colors = ['#2ca02c' if x >= 0 else '#d62728' for x in daily_chg.values]
    plt.bar(daily_chg.index, daily_chg.values, color=colors)
    plt.axhline(target_per_day, color='orange', linestyle='--', linewidth=1, label=f'Target {target_per_day:.0f}')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title('Daily PnL')
    plt.ylabel('$')
    plt.legend()
    plt.tight_layout()
    if os.environ.get("BACKTEST_SAVE_PLOTS", "").strip():
        try:
            plt.savefig("daily_pnl.png", dpi=150)
            print("Saved daily PnL plot to daily_pnl.png")
        except Exception as e:
            print(f"Could not save daily PnL plot: {e}")
        finally:
            plt.close()
    else:
        plt.show()


def main():
    tickers_env = os.environ.get("BACKTEST_TICKERS") or os.environ.get("TOPSTEP_MICRO_TICKERS")
    if tickers_env:
        tickers = [t.strip() for t in tickers_env.split(",") if t.strip()]
    else:
        tickers = [TICKER]

    all_dfs = []
    all_equities = []
    for tkr in tickers:
        try:
            print(f"Downloading data for {tkr}...")
            df = load_mnq_data(ticker=tkr)
            if df.empty:
                print(f"No data for {tkr}; skipping.")
                continue
            print(f"{tkr}: Got {len(df)} bars from {df.index[0]} to {df.index[-1]}")

            base_symbol = (tkr.split("=")[0] if "=" in tkr else tkr).strip().upper()
            print(f"Running backtest for {base_symbol} (1m/7d)...")
            trades, equity = backtest(df, base_symbol=base_symbol)

            # Per-ticker summary
            print(f"\n=== Summary for {base_symbol} ===")
            summarize_trades(trades)
            # Per-ticker last-N-hours CSV and execution plot
            try:
                hours_env = os.environ.get("BACKTEST_WINDOW_HOURS", "24")
                try:
                    win_hours = int(hours_env)
                except Exception:
                    win_hours = 24

                import pandas as _pd  # local alias to avoid scope issues
                start_ts = df.index.max() - _pd.Timedelta(hours=win_hours)
                df_tr_local = trades_to_df(trades)
                if not df_tr_local.empty:
                    mask = (_pd.to_datetime(df_tr_local["exit_time"]) >= start_ts)
                    df_24 = df_tr_local.loc[mask].copy()
                    out_name = f"{base_symbol.lower()}_trades_last{win_hours}h.csv"
                    try:
                        df_24.to_csv(out_name, index=False)
                        net_24 = float(_pd.to_numeric(df_24["pnl"], errors="coerce").sum())
                        n_full_24 = int((df_24.get("is_full_exit", True) == True).sum()) if "is_full_exit" in df_24.columns else int(len(df_24))
                        n_part_24 = int((df_24.get("is_full_exit", False) == False).sum()) if "is_full_exit" in df_24.columns else 0
                        print(f"Saved last {win_hours}h trades to {out_name} | Net PnL: ${net_24:.2f} | Full: {n_full_24} Partials: {n_part_24}")
                    except Exception as e:
                        print(f"Could not save {out_name}: {e}")

                plot_price_with_executions(df, trades, hours=win_hours, symbol=base_symbol)
            except Exception as e:
                print(f"Per-ticker 24h outputs failed for {base_symbol}: {e}")

            # Also run a dedicated 1-day backtest (1m/1d) per symbol
            try:
                print(f"Running 1d backtest for {base_symbol} (1m/1d)...")
                df_1d = load_mnq_data(ticker=tkr, period="1d")
                if not df_1d.empty:
                    trades_1d, equity_1d = backtest(df_1d, base_symbol=base_symbol)
                    # Save 1d trades CSV per symbol
                    try:
                        df_1d_tr = trades_to_df(trades_1d)
                        out1d = f"{base_symbol.lower()}_trades_1d.csv"
                        df_1d_tr.to_csv(out1d, index=False)
                        print(f"Saved 1d trades to {out1d}")
                    except Exception as e:
                        print(f"Could not save 1d trades for {base_symbol}: {e}")
                else:
                    print(f"No data for {base_symbol} 1d run.")
            except Exception as e:
                print(f"1d backtest failed for {base_symbol}: {e}")

            # Collect for aggregate CSV
            try:
                df_tr = trades_to_df(trades)
                df_tr["symbol"] = base_symbol
                all_dfs.append(df_tr)
            except Exception:
                pass
            all_equities.append((base_symbol, equity))
        except Exception as e:
            print(f"Error backtesting {tkr}: {e}")

    # Aggregate CSV if multiple tickers
    if all_dfs:
        try:
            import pandas as pd  # type: ignore
            df_all = pd.concat(all_dfs, ignore_index=True)
            out = "mnq_trades_all.csv" if len(tickers) > 1 else "mnq_trades.csv"
            df_all.to_csv(out, index=False)
            print(f"Saved aggregated trades to {out}")
        except Exception as e:
            print(f"Could not save aggregated CSV: {e}")

        # Per-symbol summary table + aggregate
        try:
            import pandas as pd  # type: ignore
            df_all["pnl"] = pd.to_numeric(df_all["pnl"], errors="coerce")
            df_all_clean = df_all.dropna(subset=["pnl"])  # ignore NaNs

            def _sumstats(df_sym: pd.DataFrame) -> dict:
                pnl = df_sym["pnl"]
                wins = (pnl > 0).sum()
                losses = (pnl < 0).sum()
                total = len(df_sym)
                win_rate = (wins / total * 100.0) if total else 0.0
                return {
                    "n": total,
                    "wins": int(wins),
                    "losses": int(losses),
                    "win_rate%": round(win_rate, 1),
                    "net": round(float(pnl.sum()), 2),
                    "avg": round(float(pnl.mean()) if total else 0.0, 2),
                    "best": round(float(pnl.max()) if total else 0.0, 2),
                    "worst": round(float(pnl.min()) if total else 0.0, 2),
                }

            if "symbol" in df_all_clean.columns:
                print("\n=== Per-symbol summary ===")
                rows = []
                for sym, grp in df_all_clean.groupby("symbol"):
                    rows.append((sym, _sumstats(grp)))
                # pretty print
                for sym, s in rows:
                    print(f"{sym:>4} | n={s['n']:4d} wins={s['wins']:4d} losses={s['losses']:4d} win%={s['win_rate%']:5.1f} net={s['net']:8.2f} avg={s['avg']:6.2f} best={s['best']:7.2f} worst={s['worst']:7.2f}")

            # Aggregate summary across all symbols
            s_all = _sumstats(df_all_clean)
            print("\n=== Aggregate summary (all symbols) ===")
            print(f"n={s_all['n']} wins={s_all['wins']} losses={s_all['losses']} win%={s_all['win_rate%']:.1f} net={s_all['net']:.2f} avg={s_all['avg']:.2f} best={s_all['best']:.2f} worst={s_all['worst']:.2f}")
        except Exception as e:
            print(f"Could not compute summary table: {e}")

    # Plot equity for the first ticker only (to keep UI manageable)
    if all_equities:
        sym0, eq0 = all_equities[0]
        plot_equity(eq0)
        plot_daily_pnl(eq0, target_per_day=1000.0)

    # Per-ticker 24h artifacts handled inside loop for each symbol.


if __name__ == "__main__":
    main()
