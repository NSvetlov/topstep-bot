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
DONCHIAN_LEN = 20

# Per-ticker point values (for $ATR filtering)
PNT_VALUE = {
    "MNQ": 2.0,
    "MES": 5.0,
    "MYM": 0.5,
    "M2K": 5.0,
}

def point_value_for(symbol: str) -> float:
    return float(PNT_VALUE.get(symbol.upper(), MULTIPLIER))

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
        "long_signal": 0,
        "short_signal": 0,
    }

    # Cooldown after stop per direction (optional; kept)
    last_stop_i_long = -10**9
    last_stop_i_short = -10**9
    COOLDOWN_BARS = 0  # cooldown disabled to mirror live behavior

    # Mode hysteresis
    last_mode: Optional[str] = None  # 'TREND', 'MR', or None

    # Thresholds for regime scoring
    ADX_TREND = 25.0
    ADX_MR = 12.0
    BB_EXPAND_MULT = 1.3
    BB_SQUEEZE_MULT = 0.75
    MR_MAX_HALF_LIFE = 45.0  # minutes on base TF

    def decide_mode(row) -> Optional[str]:
        nonlocal last_mode
        # Inputs
        regime = classify_regime(row)
        adx = float(row.get("adx14") or np.nan)
        bb_bw = float(row.get("bb_bw_5") or np.nan)
        bb_bw_ma = float(row.get("bb_bw_ma_5") or np.nan)
        hl = float(row.get("half_life") or np.nan)

        trend_score = 0
        mr_score = 0
        if regime in ("UP", "DOWN"):
            trend_score += 1
        if not np.isnan(adx) and adx >= ADX_TREND:
            trend_score += 1
        if (not np.isnan(bb_bw)) and (not np.isnan(bb_bw_ma)) and bb_bw_ma != 0 and (bb_bw >= BB_EXPAND_MULT * bb_bw_ma):
            trend_score += 1

        if (not np.isnan(hl)) and hl > 0 and hl <= MR_MAX_HALF_LIFE:
            mr_score += 1
        if not np.isnan(adx) and adx <= ADX_MR:
            mr_score += 1
        if (not np.isnan(bb_bw)) and (not np.isnan(bb_bw_ma)) and bb_bw_ma != 0 and (bb_bw <= BB_SQUEEZE_MULT * bb_bw_ma):
            mr_score += 1

        mode: Optional[str]
        if trend_score - mr_score >= 1:
            mode = "TREND"
        elif mr_score - trend_score >= 1:
            mode = None  # MR disabled
        else:
            mode = last_mode  # hysteresis: keep prior mode if indecisive
        last_mode = mode
        return mode

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
                    MIN_STOP_STEP_ATR = 0.25
                    if base_stop > stop_price + MIN_STOP_STEP_ATR * atr_at_entry:
                        stop_price = base_stop
                else:
                    base_stop = entry_price + ATR_STOP_MULT * atr_at_entry
                    if base_stop < stop_price - MIN_STOP_STEP_ATR * atr_at_entry:
                        stop_price = base_stop

                # Breakeven once price moves in favor by BE_TRIGGER_ATR
                if move_fav >= BE_TRIGGER_ATR * atr_at_entry:
                    if direction == "LONG":
                        target = entry_price + BE_OFFSET_ATR * atr_at_entry
                        if target > stop_price + MIN_STOP_STEP_ATR * atr_at_entry:
                            stop_price = target
                    else:
                        target = entry_price - BE_OFFSET_ATR * atr_at_entry
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
                        is_full_exit=False,
                        mfe_points=(max(0.0, float(mfe_price - entry_price)) if entry_price is not None and mfe_price is not None and direction == "LONG" else (max(0.0, float(entry_price - mfe_price)) if entry_price is not None and mfe_price is not None else None)),
                        mfe_price_at_exit=float(mfe_price) if mfe_price is not None else None,
                    ))
                    qty = qty - part_qty
                    position = qty if direction == "LONG" else -qty
                    pos_trend_partial2_done = True

            # MR exits disabled (no MR positions created)

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
            # ATR/$ATR entry bands per symbol with fallback to global and RTH uplift
            ATR_ENTRY_MIN = 6.0
            ATR_ENTRY_MAX = 20.0
            hh = ts.hour
            mm = ts.minute
            sym = (base_symbol or BASE_SYMBOL or "MNQ").upper()
            pv_here = point_value_for(sym)
            datr = atr * pv_here
            if sym in DATR_BAND_MAP:
                lo, hi = DATR_BAND_MAP[sym]
                if not (lo <= datr <= hi):
                    continue
            elif sym in ATR_BAND_MAP:
                lo, hi = ATR_BAND_MAP[sym]
                if not (lo <= atr <= hi):
                    continue
            elif sym in DEFAULT_ATR_BANDS:
                lo, hi = DEFAULT_ATR_BANDS[sym]
                if not (lo <= atr <= hi):
                    continue
            else:
                atr_min_use = ATR_ENTRY_MIN
                if (hh > 8 or (hh == 8 and mm >= 30)) and (hh < 11 or (hh == 11 and mm == 0)):
                    atr_min_use = max(atr_min_use, 8.0)
                if not (atr_min_use <= atr <= ATR_ENTRY_MAX):
                    continue
            if np.isnan(donch_high) or np.isnan(donch_low) or np.isnan(ema_1h):
                continue

            mode = decide_mode(row)
            long_signal = False
            short_signal = False

            if mode == "TREND":
                # Early trend breakout in direction of hourly trend
                if regime == "UP" and close > donch_high:
                    long_signal = True
                elif regime == "DOWN" and close < donch_low:
                    short_signal = True
            # MR entries disabled

            if long_signal:
                stats["long_signal"] += 1
            if short_signal:
                stats["short_signal"] += 1

            if long_signal or short_signal:
                direction = "LONG" if long_signal else "SHORT"

                # Directional cooldown after recent stop
                if direction == "LONG" and (i - last_stop_i_long) < COOLDOWN_BARS:
                    continue
                if direction == "SHORT" and (i - last_stop_i_short) < COOLDOWN_BARS:
                    continue

                # Position sizing (risk-based only)
                # ATR-based stop distance: 2x ATR
                stop_points = 2.0 * atr
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
                realized_at_entry = realized_pnl
                pos_reached18 = False
                pos_reached30 = False
                daily_trades += 1
                # Track entry and max qty for analytics
                entry_qty_local = qty
                max_qty_local = qty
                pyr_count = 0

        # record equity
        equity_curve.append(STARTING_CAPITAL + realized_pnl)
        timestamps.append(ts)

    if not trades:
        print("No entries fired; debug counts:")
        for k in [
            "bars",
            "long_signal",
            "short_signal",
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


def plot_equity(equity: pd.Series):
    plt.figure(figsize=(10, 5))
    plt.plot(equity.index, equity.values)
    plt.title("Equity Curve (MNQ Trend-Following Strategy)")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.tight_layout()
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


if __name__ == "__main__":
    main()
