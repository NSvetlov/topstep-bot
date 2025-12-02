import os
import time
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime

import numpy as np

from topstep_client import TopstepClient
from mnq_backtest import (
    load_mnq_data,
    classify_regime,
    in_entry_session,
    past_force_flat,
    # sizing params
    MIN_QTY,
    MAX_CONTRACTS,
    MULTIPLIER,
    RISK_PER_TRADE,
)
# PnL widget is optional; import lazily only when enabled


def compute_entry_signal_trend(row) -> Optional[str]:
    """Return 'LONG', 'SHORT', or None based on the simplified entry rule used in backtest."""
    regime = classify_regime(row)
    close = row["Close"]
    donch_high = row.get("donch_high", np.nan)
    donch_low = row.get("donch_low", np.nan)
    if np.isnan(donch_high) or np.isnan(donch_low):
        return None

    if regime == "UP" and close > donch_high:
        return "LONG"
    if regime == "DOWN" and close < donch_low:
        return "SHORT"
    return None


def compute_entry_signal_mr(row, z_entry: float = 0.8) -> Optional[str]:
    """Mean-reversion signal using Z-score of deviation from EMA_PULL.
    LONG when Z <= -z_entry, SHORT when Z >= z_entry. Otherwise None.
    """
    z = row.get("Z", np.nan)
    if z is None or np.isnan(z):
        return None
    if z <= -float(z_entry):
        return "LONG"
    if z >= float(z_entry):
        return "SHORT"
    return None


def decide_trading_mode(
    row,
    mr_max_half_life: float,
    adx_trend: float,
    adx_mr: float,
    bb_expand_mult: float,
    bb_squeeze_mult: float,
    prev_mode: Optional[str] = None,
) -> Optional[str]:
    """Score-based decision between 'TREND' and 'MR' with hysteresis.
    - TrendScore: +1 if hourly regime is UP/DOWN; +1 if ADX>=adx_trend; +1 if bb_bw_5 >= bb_expand_mult * bb_bw_ma_5
    - MRScore: +1 if half-life<=mr_max_half_life; +1 if ADX<=adx_mr; +1 if bb_bw_5 <= bb_squeeze_mult * bb_bw_ma_5
    - If scores tie, keep prev_mode; otherwise choose higher score.
    """
    regime = classify_regime(row)
    adx = float(row.get("adx14") or np.nan)
    bb_bw = float(row.get("bb_bw_5") or np.nan)
    bb_bw_ma = float(row.get("bb_bw_ma_5") or np.nan)
    hl = float(row.get("half_life") or np.nan)

    trend_score = 0
    mr_score = 0
    if regime in ("UP", "DOWN"):
        trend_score += 1
    if not np.isnan(adx) and adx >= adx_trend:
        trend_score += 1
    if (not np.isnan(bb_bw)) and (not np.isnan(bb_bw_ma)) and bb_bw_ma != 0 and (bb_bw >= bb_expand_mult * bb_bw_ma):
        trend_score += 1

    if (not np.isnan(hl)) and hl > 0 and hl <= mr_max_half_life:
        mr_score += 1
    if not np.isnan(adx) and adx <= adx_mr:
        mr_score += 1
    if (not np.isnan(bb_bw)) and (not np.isnan(bb_bw_ma)) and bb_bw_ma != 0 and (bb_bw <= bb_squeeze_mult * bb_bw_ma):
        mr_score += 1

    if trend_score > mr_score:
        return "TREND"
    if mr_score > trend_score:
        return "MR"
    return prev_mode


def compute_qty(row) -> Optional[int]:
    """Risk-based sizing only: qty from 2x ATR stop and RISK_PER_TRADE.
    Clamped to [MIN_QTY, MAX_CONTRACTS].
    """
    atr = row.get("atr", np.nan)
    if atr is None or np.isnan(atr) or atr <= 0:
        return None
    stop_points = 2.0 * atr
    risk_per_contract = stop_points * MULTIPLIER
    if risk_per_contract <= 0:
        return None

    qty = int(RISK_PER_TRADE / risk_per_contract)
    qty = max(qty, int(MIN_QTY))
    qty = min(qty, int(MAX_CONTRACTS))
    return qty if qty > 0 else None


def main():
    # Read required runtime config
    tickers_csv = os.environ.get("TOPSTEP_MICRO_TICKERS", "MNQ=F,MES=F,MYM=F,M2K=F").strip()
    tickers: List[str] = [t.strip() for t in tickers_csv.split(",") if t.strip()]
    if not tickers:
        raise RuntimeError("No micro tickers specified. Set TOPSTEP_MICRO_TICKERS.")

    # Derive base symbols used for contract resolution (e.g., 'MNQ' from 'MNQ=F')
    base_symbols: List[str] = []
    for t in tickers:
        base = t.split("=")[0].strip() or t
        base_symbols.append(base)

    # Optional contract live flag for available contracts lookup
    live_flag = os.environ.get("TOPSTEP_CONTRACT_LIVE", "false").strip().lower() in ("1", "true", "yes")
    # Optional explicit contract ids as CSV mapping in same order as tickers
    contract_ids_csv = os.environ.get("TOPSTEP_CONTRACT_IDS", "").strip()
    explicit_contract_ids: List[str] = [c.strip() for c in contract_ids_csv.split(",")] if contract_ids_csv else []

    # Prepare client from env and resolve IDs once
    client = TopstepClient.from_env()

    account_id = os.environ.get("TOPSTEP_ACCOUNT_ID", "").strip() or None
    if not account_id:
        try:
            account_id = client.resolve_account_id()
        except Exception as e:
            print(f"Auth/account error resolving accountId: {e}. Check TOPSTEP_API_KEY and base URL.")
            return

    # Resolve contract IDs per base symbol, allowing explicit overrides
    symbol_to_contract: Dict[str, str] = {}
    for idx, base in enumerate(base_symbols):
        cid = explicit_contract_ids[idx] if idx < len(explicit_contract_ids) and explicit_contract_ids[idx] else None
        if not cid:
            try:
                cid = client.resolve_contract_id(symbol=base, live=live_flag)
            except Exception as e:
                print(f"Contract resolution failed for symbol {base}: {e}.")
                continue
        symbol_to_contract[base] = cid

    # Reverse lookup: contract -> base symbol
    contract_to_symbol: Dict[str, str] = {v: k for k, v in symbol_to_contract.items()}

    # Tick value per micro index point (approximate). Default to MNQ if unknown.
    PNT_VALUE: Dict[str, float] = {
        "MNQ": 2.0,
        "MES": 5.0,
        "MYM": 0.5,
        "M2K": 5.0,
    }

    def point_value_for(symbol: str) -> float:
        return float(PNT_VALUE.get(symbol.upper(), MULTIPLIER))

    # Tick sizes for common micro index futures (approximate for TP price calc)
    TICK_SIZE: Dict[str, float] = {
        "MNQ": 0.25,
        "MES": 0.25,
        "MYM": 1.0,
        "M2K": 0.1,
    }

    def tick_size_for(symbol: str) -> float:
        return float(TICK_SIZE.get(symbol.upper(), 0.25))

    # ---- Per-ticker ATR/Dollar-ATR entry band parsing ----
    def _parse_band_map(raw: str) -> Dict[str, Tuple[float, float]]:
        m: Dict[str, Tuple[float, float]] = {}
        for seg in raw.split(";"):
            seg = seg.strip()
            if not seg:
                continue
            if ":" not in seg or "-" not in seg:
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
                continue
        return m

    atr_band_map: Dict[str, Tuple[float, float]] = _parse_band_map(os.environ.get("TOPSTEP_ATR_ENTRY_BANDS", ""))
    # Built-in ATR defaults (points) for specific symbols
    DEFAULT_ATR_BANDS: Dict[str, Tuple[float, float]] = {
        "MNQ": (10.0, 20.0),
    }
    datr_band_map_env: Dict[str, Tuple[float, float]] = _parse_band_map(os.environ.get("TOPSTEP_DATR_ENTRY_BANDS", ""))
    # Lower, sensible default $ATR bands if none provided via env
    # Built-in lower defaults (overridden by TOPSTEP_DATR_ENTRY_BANDS if set)
    DEFAULT_DATR_BANDS: Dict[str, Tuple[float, float]] = {
        "MNQ": (6.0, 30.0),
        "MES": (5.0, 25.0),
        "MYM": (2.0, 10.0),
        "M2K": (1.0, 10.0),
    }
    # Merge env over defaults
    datr_band_map: Dict[str, Tuple[float, float]] = {**DEFAULT_DATR_BANDS, **datr_band_map_env}

    # Track open positions we initiated: contractId -> record dict
    # record keys: side("LONG"/"SHORT"), qty(int), entry_price(float), entry_time(datetime), atr(float), mfe_price(float), exit_sent(bool)
    open_positions_local: Dict[str, Dict[str, Any]] = {}

    # Exit management settings (env overrides)
    def _env_float(name: str, default: float) -> float:
        try:
            v = float(os.environ.get(name, str(default)).strip())
            return v
        except Exception:
            return default

    def _env_int(name: str, default: int) -> int:
        try:
            v = int(float(os.environ.get(name, str(default)).strip()))
            return v
        except Exception:
            return default

    def _env_bool(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in ("1", "true", "yes", "on")

    ATR_STOP_MULT = _env_float("TOPSTEP_ATR_STOP_MULT", 1.25)
    BE_TRIGGER_ATR = _env_float("TOPSTEP_BE_TRIGGER_ATR", 1.5)
    BE_OFFSET_ATR = _env_float("TOPSTEP_BE_OFFSET_ATR", 0.10)
    TRAIL1_TRIGGER_ATR = _env_float("TOPSTEP_TRAIL1_TRIGGER_ATR", 2.5)
    TRAIL1_MULT = _env_float("TOPSTEP_TRAIL1_MULT", 2.0)
    TRAIL2_TRIGGER_ATR = _env_float("TOPSTEP_TRAIL2_TRIGGER_ATR", 4.0)
    TRAIL2_MULT = _env_float("TOPSTEP_TRAIL2_MULT", 1.5)
    TRAIL3_TRIGGER_ATR = _env_float("TOPSTEP_TRAIL3_TRIGGER_ATR", 5.0)
    TRAIL3_MULT = _env_float("TOPSTEP_TRAIL3_MULT", 1.25)
    RATCHET_TRIGGER_ATR = _env_float("TOPSTEP_RATCHET_TRIGGER_ATR", 3.0)
    RATCHET_OFFSET_ATR = _env_float("TOPSTEP_RATCHET_OFFSET_ATR", 0.5)

    # Stop behavior tuning
    STOP_CLOSE_ONLY = _env_bool("TOPSTEP_STOP_CLOSE_ONLY", True)
    STOP_CLOSE_CONFIRMED = _env_int("TOPSTEP_STOP_CLOSE_CONFIRMED", 1)
    try:
        STOP_TICK_BUFFER = float(os.environ.get("TOPSTEP_STOP_TICK_BUFFER", "2").strip())
    except Exception:
        STOP_TICK_BUFFER = 2.0
    MIN_STOP_STEP_ATR = _env_float("TOPSTEP_MIN_STOP_STEP_ATR", 0.25)

    # Pyramiding
    PYRAMID_ENABLED = _env_bool("TOPSTEP_PYRAMID_ENABLED", True)
    PYRAMID_STEP_ATR = _env_float("TOPSTEP_PYRAMID_STEP_ATR", 1.5)
    PYRAMID_ADD_QTY = _env_int("TOPSTEP_PYRAMID_ADD_QTY", 1)
    TIME_STOP_MIN = _env_int("TOPSTEP_TIME_STOP_MIN", 90)
    ENFORCE_FORCE_FLAT = _env_bool("TOPSTEP_FORCE_FLAT", True)

    # PnL accounting and optional UI widget
    realized_pnl: float = 0.0
    peak_equity: float = 0.0
    max_dd: float = 0.0
    wins = 0
    losses = 0
    acct_label = os.environ.get("TOPSTEP_ACCOUNT_LABEL") or os.environ.get("TOPSTEP_ACCOUNT_ID") or ""
    title = f"Topstep PnL{(' - ' + acct_label) if acct_label else ''}"
    # Control widget via env flag (default: disabled)
    show_widget_env = os.environ.get("TOPSTEP_SHOW_WIDGET", "false").strip().lower()
    show_widget = show_widget_env in ("1", "true", "yes")
    widget = None
    if show_widget:
        try:
            from pnl_widget import PnLWidget  # lazy import to avoid matplotlib if disabled
            widget = PnLWidget(title=title)
        except Exception as _e:
            # If widget fails to init, continue without UI
            widget = None

    # Brackets default disabled; env overrides allowed
    use_bracket_env = os.environ.get("TOPSTEP_USE_BRACKET", "false").strip().lower()
    use_bracket = use_bracket_env in ("1", "true", "yes")
    stop_ticks_env = os.environ.get("TOPSTEP_STOP_TICKS", "10").strip()
    take_ticks_env = os.environ.get("TOPSTEP_TAKE_TICKS", "20").strip()
    stop_ticks_val: Optional[int] = int(stop_ticks_env) if stop_ticks_env.isdigit() else 10
    take_ticks_val: Optional[int] = int(take_ticks_env) if take_ticks_env.isdigit() else 20
    if not use_bracket:
        stop_ticks_val = None
        take_ticks_val = None

    # Live trade CSV log (best-effort)
    log_path = os.environ.get("TOPSTEP_LIVE_TRADE_LOG", "live_trades.csv").strip() or "live_trades.csv"
    _log_header_written = False

    def _log_trade(event: str, data: Dict[str, Any]) -> None:
        nonlocal _log_header_written
        fields = [
            "ts",
            "event",
            "account_id",
            "symbol",
            "contract_id",
            "side",
            "qty",
            "entry_price",
            "init_stop",
            "tp_price",
            "tp_ticks",
            "tp_spec",
            "mode",
            "atr_at_entry",
            "exit_price",
            "realized_pnl",
            "reason",
        ]
        try:
            write_header = (not _log_header_written) and (not os.path.exists(log_path))
            with open(log_path, mode="a", newline="", encoding="utf-8") as f:
                import csv as _csv
                w = _csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                if write_header:
                    w.writeheader()
                row = {"event": event, **data}
                w.writerow(row)
            _log_header_written = True
        except Exception:
            pass

    # Main 1-minute polling loop (no entry session restriction)
    # Regime thresholds (env overrides)
    adx_trend = float(os.environ.get("TOPSTEP_ADX_TREND", "25").strip() or 25)
    adx_mr = float(os.environ.get("TOPSTEP_ADX_MR", "12").strip() or 12)
    bb_expand_mult = float(os.environ.get("TOPSTEP_BB_EXPAND_MULT", "1.3").strip() or 1.3)
    bb_squeeze_mult = float(os.environ.get("TOPSTEP_BB_SQUEEZE_MULT", "0.75").strip() or 0.75)

    # Keep previous mode per symbol to reduce flapping
    prev_mode_by_symbol: Dict[str, Optional[str]] = {}

    while True:
        try:
            ts = None
            last_price: Dict[str, float] = {}
            last_ind: Dict[str, Dict[str, float]] = {}
            # Track previous ADX per symbol to detect roll-over
            if 'prev_adx_by_symbol' not in globals():
                # create once
                globals()['prev_adx_by_symbol'] = {}
            prev_adx_by_symbol: Dict[str, float] = globals()['prev_adx_by_symbol']  # type: ignore[assignment]
            # Directional cooldown removed: trade decisions are immediate based on signals
            # Process each micro ticker independently
            for tkr, base in zip(tickers, base_symbols):
                if base not in symbol_to_contract:
                    continue
                contract_id = symbol_to_contract[base]

                df = load_mnq_data(ticker=tkr)
                if df.empty:
                    print(f"{tkr}: no data loaded.")
                    continue
                row = df.iloc[-1]
                ts = row.name
                try:
                    last_price[base] = float(row["Close"])  # cache for PnL update
                except Exception:
                    pass
                # Cache indicators used for exits (e.g., Z for MR)
                try:
                    last_ind[base] = {
                        "Z": float(row.get("Z") if "Z" in row else np.nan),
                        "ema_1h_50": float(row.get("ema_1h_50") if "ema_1h_50" in row else np.nan),
                        "atr": float(row.get("atr") if "atr" in row else np.nan),
                        "adx14": float(row.get("adx14") if "adx14" in row else np.nan),
                    }
                except Exception:
                    last_ind[base] = {"Z": np.nan, "ema_1h_50": np.nan, "atr": np.nan, "adx14": np.nan}
                # Capture ADX roll-over input
                try:
                    adx_now = last_ind[base]["adx14"]
                    last_ind[base]["adx14_prev"] = prev_adx_by_symbol.get(base, np.nan)
                    if not np.isnan(adx_now):
                        prev_adx_by_symbol[base] = adx_now
                except Exception:
                    pass

                # In-position guard via open positions (per contract)
                try:
                    open_pos = client.search_open_positions(account_id=account_id)
                except Exception as e:
                    print(f"{ts}: could not fetch open positions: {e}")
                    open_pos = []

                in_market = False
                try:
                    # Normalize to iterable
                    if isinstance(open_pos, dict):
                        items = open_pos.get("positions") or open_pos.get("data") or open_pos.get("items") or []
                    else:
                        items = open_pos
                    for p in items:
                        cid = str(p.get("contractId") or p.get("contractID") or p.get("contract_id") or "")
                        qty = p.get("quantity") or p.get("qty") or p.get("netPosition") or 0
                        if cid and cid == str(contract_id):
                            try:
                                if int(qty) != 0:
                                    in_market = True
                                    break
                            except Exception:
                                in_market = True
                                break
                except Exception:
                    in_market = True

                if in_market:
                    print(f"{ts}: already in market on {base}; no new entry.")
                    continue

                # Select strategy based on regime/half-life (with RTH uplift for ADX trend)
                mr_hl_env = os.environ.get("TOPSTEP_MR_MAX_HALF_LIFE", "45").strip()
                try:
                    mr_hl = float(mr_hl_env)
                except Exception:
                    mr_hl = 45.0
                prev_mode = prev_mode_by_symbol.get(base)
                # Time-of-day windows
                try:
                    hh = int(ts.hour)
                    mm = int(ts.minute)
                except Exception:
                    hh = datetime.now().hour
                    mm = datetime.now().minute
                adx_trend_local = adx_trend
                if (hh > 8 or (hh == 8 and mm >= 30)) and (hh < 11 or (hh == 11 and mm == 0)):
                    adx_trend_local = max(adx_trend_local, 30.0)
                mode = decide_trading_mode(
                    row,
                    mr_max_half_life=mr_hl,
                    adx_trend=adx_trend_local,
                    adx_mr=adx_mr,
                    bb_expand_mult=bb_expand_mult,
                    bb_squeeze_mult=bb_squeeze_mult,
                    prev_mode=prev_mode,
                )
                # Disable MR trades: ignore MR mode
                if mode == "MR":
                    mode = None
                prev_mode_by_symbol[base] = mode

                z_entry_env = os.environ.get("TOPSTEP_MR_Z_ENTRY", "1.0").strip()
                try:
                    z_entry = float(z_entry_env)
                except Exception:
                    z_entry = 1.0
                # Midday uplift 12:00-14:00
                if (hh > 12 or (hh == 12 and mm >= 0)) and (hh < 14 or (hh == 14 and mm == 0)):
                    z_entry = max(z_entry, 1.2)

                # ATR/$ATR entry gating disabled per request (kept ATR must be > 0 check below)

                side = None
                if mode == "TREND":
                    side = compute_entry_signal_trend(row)
                # MR entries disabled
                # No cooldown guard: proceed if signal present
                if not side:
                    print(f"{ts}: no entry signal for {base} (mode={mode or 'NONE'}).")
                    continue

                # Risk-based sizing per symbol
                try:
                    atr_here = float(row.get("atr") or 0.0)
                except Exception:
                    atr_here = 0.0
                if atr_here <= 0:
                    print(f"{ts}: no valid ATR for {base}; skip.")
                    continue
                stop_points = 2.0 * atr_here
                pv = point_value_for(base)
                risk_per_contract = stop_points * pv
                if risk_per_contract <= 0:
                    print(f"{ts}: invalid risk per contract for {base}; skip.")
                    continue
                qty = int(RISK_PER_TRADE / risk_per_contract)
                qty = max(qty, int(MIN_QTY))
                qty = min(qty, int(MAX_CONTRACTS))
                if not qty:
                    print(f"{ts}: no valid qty for {base}; skip.")
                    continue

                try:
                    # Compute initial stop for logging based on ATR and configured multiple
                    atr_here = float(row.get("atr") or 0.0)
                    entry_px = float(row["Close"]) if "Close" in row else None
                    init_stop = None
                    if atr_here and entry_px:
                        stop_dist = ATR_STOP_MULT * atr_here
                        init_stop = entry_px - stop_dist if side == "LONG" else entry_px + stop_dist
                    resp = client.place_market_order(
                        account_id=account_id,
                        contract_id=contract_id,
                        side=("BUY" if side == "LONG" else "SELL"),
                        size=qty,
                        stop_ticks=stop_ticks_val,
                        take_ticks=take_ticks_val,
                    )
                    # Remember last row for exit logic by symbol
                    last_price[base] = float(row["Close"]) if "Close" in row else last_price.get(base, None)

                    # If broker rejects bracket fields, retry without brackets
                    if isinstance(resp, dict) and not resp.get("success", True):
                        msg = str(resp.get("errorMessage", "")).lower()
                        if "brackets" in msg and "auto oco" in msg:
                            try:
                                resp2 = client.place_market_order(
                                    account_id=account_id,
                                    contract_id=contract_id,
                                    side=("BUY" if side == "LONG" else "SELL"),
                                    size=qty,
                                    stop_ticks=None,
                                    take_ticks=None,
                                )
                                print(f"{ts}: bracket rejected; retried without brackets -> {resp2}")
                                # Track local position on assumed fill
                                if contract_id not in open_positions_local:
                                    open_positions_local[contract_id] = {
                                        "side": side,
                                        "qty": int(qty),
                                        "entry_price": float(row["Close"]),
                                        "entry_time": datetime.now(),
                                        "atr": float(row.get("atr") or 0.0),
                                        "mode": mode or "TREND",
                                        "init_stop": float(init_stop) if init_stop is not None else None,
                                        "mfe_price": float(row["Close"]),
                                        "exit_sent": False,
                                    }
                            except Exception as e2:
                                print(f"{ts}: retry without brackets failed: {e2}")
                        else:
                            print(f"{ts}: order response indicates failure -> {resp}")
                    else:
                        # Determine take-profit specification for printing
                        tp_price = None
                        tp_ticks = None
                        tp_spec = None
                        if use_bracket and take_ticks_val is not None and entry_px is not None:
                            try:
                                tsz = tick_size_for(base)
                                tp_ticks = int(take_ticks_val)
                                tp_price = (entry_px + tp_ticks * tsz) if side == "LONG" else (entry_px - tp_ticks * tsz)
                            except Exception:
                                tp_price = None
                        else:
                            # Dynamic exits: summarize plan
                            if mode == "MR":
                                z_exit_env = os.environ.get("TOPSTEP_MR_Z_EXIT", "0.1").strip()
                                tp_spec = f"MR Z→{z_exit_env}"
                            else:
                                tp_spec = "dynamic trail"

                        # Success path: print one consolidated message
                        base_msg = f"{ts}: submitted {side} {qty} {base}"
                        if entry_px is not None:
                            base_msg += f" @ {entry_px:.2f}"
                        try:
                            base_msg += f" | ATR {atr_here:.2f}"
                        except Exception:
                            pass
                        if init_stop is not None:
                            base_msg += f" | init stop {init_stop:.2f}"
                        if tp_price is not None:
                            base_msg += f" | tp {tp_price:.2f}"
                        elif tp_spec is not None:
                            base_msg += f" | tp {tp_spec}"
                        base_msg += f" | mode {mode} (acct {account_id}, contract {contract_id}) -> {resp}"
                        print(base_msg)
                        # Track local position on assumed fill
                        if contract_id not in open_positions_local:
                            open_positions_local[contract_id] = {
                                "side": side,
                                "qty": int(qty),
                                "entry_price": float(row["Close"]),
                                "entry_time": datetime.now(),
                                "atr": float(row.get("atr") or 0.0),
                                "mode": mode or "TREND",
                                "init_stop": float(init_stop) if init_stop is not None else None,
                                "mfe_price": float(row["Close"]),
                                "exit_sent": False,
                            }
                        # Log entry
                        _log_trade(
                            "ENTRY",
                            {
                                "ts": ts,
                                "account_id": account_id,
                                "symbol": base,
                                "contract_id": contract_id,
                                "side": side,
                                "qty": int(qty),
                                "entry_price": float(entry_px) if entry_px is not None else None,
                                "init_stop": float(init_stop) if init_stop is not None else None,
                                "tp_price": float(tp_price) if tp_price is not None else None,
                                "tp_ticks": int(tp_ticks) if tp_ticks is not None else None,
                                "tp_spec": tp_spec,
                                "mode": mode,
                                "atr_at_entry": float(atr_here) if atr_here is not None else None,
                            },
                        )
                except Exception as e:
                    print(f"{ts}: order submit failed for {base}: {e}")

            # --- After iterating all symbols: compute PnL and update widget ---
            now_dt = datetime.now()

            # Refresh open positions once for closure detection and to avoid stale local state
            try:
                open_pos_all = client.search_open_positions(account_id=account_id)
            except Exception:
                open_pos_all = []

            active_contracts: Dict[str, int] = {}
            try:
                if isinstance(open_pos_all, dict):
                    items = open_pos_all.get("positions") or open_pos_all.get("data") or open_pos_all.get("items") or []
                else:
                    items = open_pos_all
                for p in items:
                    cid = str(p.get("contractId") or p.get("contractID") or p.get("contract_id") or "")
                    qty = p.get("quantity") or p.get("qty") or p.get("netPosition") or 0
                    try:
                        qty_int = int(qty)
                    except Exception:
                        qty_int = 0
                    active_contracts[cid] = qty_int
            except Exception:
                pass

            # Compute exits and unrealized PnL across all locally tracked positions
            unrealized_total = 0.0
            to_remove: List[str] = []
            for cid, rec in list(open_positions_local.items()):
                base = contract_to_symbol.get(cid)
                if not base:
                    continue

                # Get latest price from last fetched data for that symbol
                # Prefer the last price observed in this cycle; fall back to fetch if missing
                last_close: Optional[float] = last_price.get(base)
                if last_close is None:
                    try:
                        tkr = next((t for t in tickers if t.split("=")[0] == base), None)
                        if not tkr:
                            continue
                        df_latest = load_mnq_data(ticker=tkr)
                        if df_latest.empty:
                            continue
                        last_close = float(df_latest.iloc[-1]["Close"])
                    except Exception:
                        continue

                p_side: str = str(rec.get("side"))
                p_qty: int = int(rec.get("qty", 0))
                p_entry: float = float(rec.get("entry_price", last_close))
                p_time: datetime = rec.get("entry_time") or datetime.now()
                atr_entry: float = float(rec.get("atr") or 0.0)
                mode_rec: str = str(rec.get("mode") or "TREND")
                mfe_price: float = float(rec.get("mfe_price") or p_entry)
                exit_sent: bool = bool(rec.get("exit_sent", False))

                # Update MFE price
                if p_side == "LONG":
                    mfe_price = max(mfe_price, last_close)
                else:
                    mfe_price = min(mfe_price, last_close)
                rec["mfe_price"] = mfe_price

                # Pyramiding: add on favorable moves at ATR steps
                if PYRAMID_ENABLED and atr_entry > 0 and p_qty < MAX_CONTRACTS:
                    try:
                        pyr_count = int(rec.get("pyr_count", 0))
                    except Exception:
                        pyr_count = 0
                    next_level = (pyr_count + 1) * PYRAMID_STEP_ATR * atr_entry
                    move_fav_now = (last_close - p_entry) if p_side == "LONG" else (p_entry - last_close)
                    if move_fav_now >= next_level:
                        add_qty = min(PYRAMID_ADD_QTY, MAX_CONTRACTS - p_qty)
                        if add_qty > 0:
                            try:
                                add_side = "BUY" if p_side == "LONG" else "SELL"
                                resp_add = client.place_market_order(
                                    account_id=account_id,
                                    contract_id=cid,
                                    side=add_side,
                                    size=int(add_qty),
                                    stop_ticks=None,
                                    take_ticks=None,
                                )
                                rec["qty"] = p_qty + add_qty
                                rec["pyr_count"] = pyr_count + 1
                                print(f"{now_dt}: PYRAMID {add_side} {add_qty} {base} (contract {cid}) -> {resp_add}")
                                # refresh p_qty for downstream calc
                                p_qty = rec["qty"]
                                _log_trade("ADD", {
                                    "ts": now_dt,
                                    "account_id": account_id,
                                    "symbol": base,
                                    "contract_id": cid,
                                    "side": p_side,
                                    "qty": int(add_qty),
                                    "entry_price": float(p_entry),
                                    "reason": "PYRAMID",
                                    "mode": mode_rec,
                                })
                            except Exception:
                                pass

                # Exit rules (only if position is live and we haven't already sent an exit)
                qty_live = int(active_contracts.get(cid, 0))
                should_exit = False
                if qty_live > 0 and not exit_sent:
                    # Time stop (adaptive: extend if progress >= 1x ATR)
                    minutes_in_trade = (now_dt - p_time).total_seconds() / 60.0
                    try:
                        TIME_STOP_EXT_MIN = int(float(os.environ.get("TOPSTEP_TIME_STOP_EXT_MIN", "60").strip()))
                    except Exception:
                        TIME_STOP_EXT_MIN = 60
                    eff_time_stop = TIME_STOP_MIN
                    try:
                        move_fav_ts = (last_close - p_entry) if p_side == "LONG" else (p_entry - last_close)
                        if atr_entry and move_fav_ts >= 1.0 * atr_entry:
                            eff_time_stop = TIME_STOP_MIN + TIME_STOP_EXT_MIN
                    except Exception:
                        pass
                    if TIME_STOP_MIN > 0 and minutes_in_trade >= eff_time_stop:
                        should_exit = True

                    # Force flat window
                    if not should_exit and ENFORCE_FORCE_FLAT and past_force_flat(now_dt):
                        should_exit = True

                    # Additional MR exit: take profits on Z reversion toward zero
                    if not should_exit and mode_rec == "MR":
                        try:
                            z_val = float(last_ind.get(base, {}).get("Z", np.nan))
                        except Exception:
                            z_val = np.nan
                        z_exit_env = os.environ.get("TOPSTEP_MR_Z_EXIT", "0.1").strip()
                        try:
                            z_exit = float(z_exit_env)
                        except Exception:
                            z_exit = 0.1
                        if not np.isnan(z_val):
                            if p_side == "LONG" and z_val >= z_exit:
                                should_exit = True
                            elif p_side == "SHORT" and z_val <= -z_exit:
                                should_exit = True

                    # MR partial at Z->threshold (default 0.5) keeps runner to MR_Z_EXIT
                    if not should_exit and mode_rec == "MR" and p_qty >= 2:
                        try:
                            z_val = float(last_ind.get(base, {}).get("Z", np.nan))
                            z_partial = float(os.environ.get("TOPSTEP_MR_Z_PARTIAL", "0.5").strip())
                        except Exception:
                            z_val = np.nan
                            z_partial = 0.5
                        partial_done = bool(rec.get("mr_partial_done", False))
                        if not partial_done and not np.isnan(z_val):
                            do_partial = (p_side == "LONG" and z_val >= z_partial) or (p_side == "SHORT" and z_val <= -z_partial)
                            if do_partial:
                                part_qty = max(1, p_qty // 3)
                                try:
                                    exit_side = "SELL" if p_side == "LONG" else "BUY"
                                    resp_px = client.place_market_order(
                                        account_id=account_id,
                                        contract_id=cid,
                                        side=exit_side,
                                        size=int(part_qty),
                                        stop_ticks=None,
                                        take_ticks=None,
                                    )
                                    rec["qty"] = p_qty - part_qty
                                    rec["mr_partial_done"] = True
                                    _log_trade("PARTIAL_EXIT", {
                                        "ts": now_dt,
                                        "account_id": account_id,
                                        "symbol": base,
                                        "contract_id": cid,
                                        "side": p_side,
                                        "qty": int(part_qty),
                                        "entry_price": float(p_entry),
                                        "exit_price": float(last_close),
                                        "reason": "MR_PARTIAL_Z",
                                        "mode": mode_rec,
                                    })
                                    print(f"{now_dt}: MR_PARTIAL_Z {exit_side} {part_qty} {base} (contract {cid}) -> {resp_px}")
                                except Exception:
                                    pass

                    # EXHAUSTION partial: allow runner; require profit + ADX roll-over
                    if not should_exit and p_qty >= 2:
                        try:
                            ema_1h_now = float(last_ind.get(base, {}).get("ema_1h_50", np.nan))
                            atr_now = float(last_ind.get(base, {}).get("atr", np.nan))
                            adx_now = float(last_ind.get(base, {}).get("adx14", np.nan))
                            adx_prev = float(last_ind.get(base, {}).get("adx14_prev", np.nan))
                            adx_rollover = (not np.isnan(adx_now)) and (not np.isnan(adx_prev)) and (adx_now < adx_prev)
                            try:
                                EXH_ATR_MULT = float(os.environ.get("TOPSTEP_EXH_ATR_MULT", "4.0").strip())
                            except Exception:
                                EXH_ATR_MULT = 4.0
                            if not np.isnan(ema_1h_now) and atr_now > 0 and adx_rollover:
                                pv = point_value_for(base)
                                unreal_now = ((last_close - p_entry) if p_side == "LONG" else (p_entry - last_close)) * p_qty * pv
                                min_profit_dlr = 0.5 * atr_now * pv * p_qty
                                trigger_long = p_side == "LONG" and last_close > ema_1h_now + EXH_ATR_MULT * atr_now
                                trigger_short = p_side == "SHORT" and last_close < ema_1h_now - EXH_ATR_MULT * atr_now
                                if (trigger_long or trigger_short) and unreal_now >= min_profit_dlr:
                                    part_qty = max(1, p_qty // 2)
                                    try:
                                        exit_side = "SELL" if p_side == "LONG" else "BUY"
                                        resp_px = client.place_market_order(
                                            account_id=account_id,
                                            contract_id=cid,
                                            side=exit_side,
                                            size=int(part_qty),
                                            stop_ticks=None,
                                            take_ticks=None,
                                        )
                                        # Log partial
                                        _log_trade(
                                            "PARTIAL_EXIT",
                                            {
                                                "ts": now_dt,
                                                "account_id": account_id,
                                                "symbol": base,
                                                "contract_id": cid,
                                                "side": p_side,
                                                "qty": int(part_qty),
                                                "entry_price": float(p_entry),
                                                "exit_price": float(last_close),
                                                "realized_pnl": None,
                                                "mode": mode_rec,
                                                "init_stop": rec.get("init_stop"),
                                                "atr_at_entry": rec.get("atr"),
                                                "reason": "EXHAUSTION_PARTIAL",
                                            },
                                        )
                                        # Update local record
                                        rec["qty"] = p_qty - part_qty
                                        rec["exit_sent"] = False
                                        print(f"{now_dt}: EXHAUSTION_PARTIAL {exit_side} {part_qty} {base} (contract {cid}) -> {resp_px}")
                                        # do not set should_exit; leave runner
                                    except Exception as _e:
                                        pass
                        except Exception:
                            pass

                    # Hard dollar stop (additional tail guard)
                    if not should_exit:
                        try:
                            HARD_DLR_STOP = float(os.environ.get("TOPSTEP_HARD_DLR_STOP", "150").strip())
                        except Exception:
                            HARD_DLR_STOP = 150.0
                        pv = point_value_for(base)
                        unreal_now = ((last_close - p_entry) if p_side == "LONG" else (p_entry - last_close)) * p_qty * pv
                        if unreal_now <= -HARD_DLR_STOP:
                            should_exit = True

                    # TREND partial at +1.5x ATR move
                    if not should_exit and mode_rec == "TREND" and p_qty >= 2 and atr_entry > 0:
                        move_fav = (last_close - p_entry) if p_side == "LONG" else (p_entry - last_close)
                        try:
                            trend_partial_mult = float(os.environ.get("TOPSTEP_TREND_PARTIAL_ATR", "1.5").strip())
                        except Exception:
                            trend_partial_mult = 1.5
                        trend_partial_done = bool(rec.get("trend_partial_done", False))
                        if not trend_partial_done and move_fav >= trend_partial_mult * atr_entry:
                            part_qty = max(1, p_qty // 3)
                            try:
                                exit_side = "SELL" if p_side == "LONG" else "BUY"
                                resp_px = client.place_market_order(
                                    account_id=account_id,
                                    contract_id=cid,
                                    side=exit_side,
                                    size=int(part_qty),
                                    stop_ticks=None,
                                    take_ticks=None,
                                )
                                rec["qty"] = p_qty - part_qty
                                rec["trend_partial_done"] = True
                                _log_trade("PARTIAL_EXIT", {
                                    "ts": now_dt,
                                    "account_id": account_id,
                                    "symbol": base,
                                    "contract_id": cid,
                                    "side": p_side,
                                    "qty": int(part_qty),
                                    "entry_price": float(p_entry),
                                    "exit_price": float(last_close),
                                    "reason": "TREND_PARTIAL_ATR",
                                    "mode": mode_rec,
                                })
                                print(f"{now_dt}: TREND_PARTIAL_ATR {exit_side} {part_qty} {base} (contract {cid}) -> {resp_px}")
                            except Exception:
                                pass

                    # TREND second partial at +3.0x ATR
                    if not should_exit and mode_rec == "TREND" and p_qty >= 2 and atr_entry > 0:
                        move_fav = (last_close - p_entry) if p_side == "LONG" else (p_entry - last_close)
                        try:
                            trend_partial2_mult = float(os.environ.get("TOPSTEP_TREND_PARTIAL2_ATR", "3.0").strip())
                        except Exception:
                            trend_partial2_mult = 3.0
                        trend_partial2_done = bool(rec.get("trend_partial2_done", False))
                        if not trend_partial2_done and move_fav >= trend_partial2_mult * atr_entry:
                            part_qty = max(1, p_qty // 3)
                            try:
                                exit_side = "SELL" if p_side == "LONG" else "BUY"
                                resp_px = client.place_market_order(
                                    account_id=account_id,
                                    contract_id=cid,
                                    side=exit_side,
                                    size=int(part_qty),
                                    stop_ticks=None,
                                    take_ticks=None,
                                )
                                rec["qty"] = p_qty - part_qty
                                rec["trend_partial2_done"] = True
                                _log_trade("PARTIAL_EXIT", {
                                    "ts": now_dt,
                                    "account_id": account_id,
                                    "symbol": base,
                                    "contract_id": cid,
                                    "side": p_side,
                                    "qty": int(part_qty),
                                    "entry_price": float(p_entry),
                                    "exit_price": float(last_close),
                                    "reason": "TREND_PARTIAL2_ATR",
                                    "mode": mode_rec,
                                })
                                print(f"{now_dt}: TREND_PARTIAL2_ATR {exit_side} {part_qty} {base} (contract {cid}) -> {resp_px}")
                            except Exception:
                                pass

                    # Price-based stops using entry ATR
                    if not should_exit and atr_entry > 0:
                        # initial stop
                        if p_side == "LONG":
                            new_stop = p_entry - ATR_STOP_MULT * atr_entry
                            # min step hysteresis
                            if stop_price is None or new_stop > stop_price + MIN_STOP_STEP_ATR * atr_entry:
                                stop_price = new_stop
                            move_fav = last_close - p_entry
                            # breakeven
                            if move_fav >= BE_TRIGGER_ATR * atr_entry:
                                target_stop = p_entry + BE_OFFSET_ATR * atr_entry
                                if stop_price is None or target_stop > stop_price + MIN_STOP_STEP_ATR * atr_entry:
                                    stop_price = max(stop_price or target_stop, target_stop)
                            # trails
                            if move_fav >= TRAIL1_TRIGGER_ATR * atr_entry:
                                target_stop = mfe_price - TRAIL1_MULT * atr_entry
                                if stop_price is None or target_stop > stop_price + MIN_STOP_STEP_ATR * atr_entry:
                                    stop_price = max(stop_price or target_stop, target_stop)
                            if move_fav >= TRAIL2_TRIGGER_ATR * atr_entry:
                                target_stop = mfe_price - TRAIL2_MULT * atr_entry
                                if stop_price is None or target_stop > stop_price + MIN_STOP_STEP_ATR * atr_entry:
                                    stop_price = max(stop_price or target_stop, target_stop)
                            if move_fav >= TRAIL3_TRIGGER_ATR * atr_entry:
                                target_stop = mfe_price - TRAIL3_MULT * atr_entry
                                if stop_price is None or target_stop > stop_price + MIN_STOP_STEP_ATR * atr_entry:
                                    stop_price = max(stop_price or target_stop, target_stop)
                            if move_fav >= RATCHET_TRIGGER_ATR * atr_entry:
                                target_stop = p_entry + RATCHET_OFFSET_ATR * atr_entry
                                if stop_price is None or target_stop > stop_price + MIN_STOP_STEP_ATR * atr_entry:
                                    stop_price = max(stop_price or target_stop, target_stop)
                            # Close-only stop with buffer
                            eff_stop = stop_price
                            if eff_stop is not None:
                                eff_stop = eff_stop - STOP_TICK_BUFFER * tick_size_for(base)
                            # Confirmation counter
                            if STOP_CLOSE_ONLY and eff_stop is not None:
                                c = int(rec.get("stop_breach", 0))
                                if last_close <= eff_stop:
                                    c += 1
                                    if c >= STOP_CLOSE_CONFIRMED:
                                        should_exit = True
                                else:
                                    c = 0
                                rec["stop_breach"] = c
                            elif stop_price is not None and last_close <= stop_price:
                                should_exit = True
                        else:  # SHORT
                            new_stop = p_entry + ATR_STOP_MULT * atr_entry
                            if stop_price is None or new_stop < stop_price - MIN_STOP_STEP_ATR * atr_entry:
                                stop_price = new_stop
                            move_fav = p_entry - last_close
                            if move_fav >= BE_TRIGGER_ATR * atr_entry:
                                target_stop = p_entry - BE_OFFSET_ATR * atr_entry
                                if stop_price is None or target_stop < stop_price - MIN_STOP_STEP_ATR * atr_entry:
                                    stop_price = min(stop_price or target_stop, target_stop)
                            if move_fav >= TRAIL1_TRIGGER_ATR * atr_entry:
                                target_stop = mfe_price + TRAIL1_MULT * atr_entry
                                if stop_price is None or target_stop < stop_price - MIN_STOP_STEP_ATR * atr_entry:
                                    stop_price = min(stop_price or target_stop, target_stop)
                            if move_fav >= TRAIL2_TRIGGER_ATR * atr_entry:
                                target_stop = mfe_price + TRAIL2_MULT * atr_entry
                                if stop_price is None or target_stop < stop_price - MIN_STOP_STEP_ATR * atr_entry:
                                    stop_price = min(stop_price or target_stop, target_stop)
                            if move_fav >= TRAIL3_TRIGGER_ATR * atr_entry:
                                target_stop = mfe_price + TRAIL3_MULT * atr_entry
                                if stop_price is None or target_stop < stop_price - MIN_STOP_STEP_ATR * atr_entry:
                                    stop_price = min(stop_price or target_stop, target_stop)
                            if move_fav >= RATCHET_TRIGGER_ATR * atr_entry:
                                target_stop = p_entry - RATCHET_OFFSET_ATR * atr_entry
                                if stop_price is None or target_stop < stop_price - MIN_STOP_STEP_ATR * atr_entry:
                                    stop_price = min(stop_price or target_stop, target_stop)
                            eff_stop = stop_price
                            if eff_stop is not None:
                                eff_stop = eff_stop + STOP_TICK_BUFFER * tick_size_for(base)
                            if STOP_CLOSE_ONLY and eff_stop is not None:
                                c = int(rec.get("stop_breach", 0))
                                if last_close >= eff_stop:
                                    c += 1
                                    if c >= STOP_CLOSE_CONFIRMED:
                                        should_exit = True
                                else:
                                    c = 0
                                rec["stop_breach"] = c
                            elif stop_price is not None and last_close >= stop_price:
                                should_exit = True

                # Send exit order if needed
                if qty_live > 0 and should_exit and not exit_sent:
                    try:
                        exit_side = "SELL" if p_side == "LONG" else "BUY"
                        resp_exit = client.place_market_order(
                            account_id=account_id,
                            contract_id=cid,
                            side=exit_side,
                            size=int(qty_live),
                            stop_ticks=None,
                            take_ticks=None,
                        )
                        rec["exit_sent"] = True
                        print(f"{now_dt}: exit sent {exit_side} {qty_live} {base} (contract {cid}) -> {resp_exit}")
                    except Exception as e:
                        print(f"{now_dt}: exit order failed for {base}: {e}")

                # PnL calc (unrealized)
                pv = point_value_for(base)
                if p_side == "LONG":
                    unreal = (last_close - p_entry) * p_qty * pv
                else:
                    unreal = (p_entry - last_close) * p_qty * pv
                unrealized_total += unreal

                # Detect closure (position no longer active)
                qty_live = active_contracts.get(cid, 0)
                if qty_live == 0:
                    # Realize PnL using current price approximation
                    if p_side == "LONG":
                        realized = (last_close - p_entry) * p_qty * pv
                    else:
                        realized = (p_entry - last_close) * p_qty * pv
                    realized_pnl += realized
                    if realized >= 0:
                        wins += 1
                    else:
                        losses += 1
                    # Log full exit
                    _log_trade(
                        "EXIT",
                        {
                            "ts": now_dt,
                            "account_id": account_id,
                            "symbol": base,
                            "contract_id": cid,
                            "side": p_side,
                            "qty": int(p_qty),
                            "entry_price": float(p_entry),
                            "exit_price": float(last_close),
                            "realized_pnl": float(realized),
                            "mode": str(rec.get("mode") or "TREND"),
                            "init_stop": rec.get("init_stop"),
                            "atr_at_entry": rec.get("atr"),
                            "reason": "CLOSED",
                        },
                    )
                    # No post-exit cooldown
                    to_remove.append(cid)
                elif qty_live != p_qty and qty_live > 0:
                    # Partial close: realize PnL on reduced size
                    delta = p_qty - qty_live
                    if delta > 0:
                        if p_side == "LONG":
                            realized = (last_close - p_entry) * delta * pv
                        else:
                            realized = (p_entry - last_close) * delta * pv
                        realized_pnl += realized
                        # Update stored qty to live qty
                        rec["qty"] = qty_live
                        # Allow subsequent exit signals to send another order
                        rec["exit_sent"] = False
                        # Log partial exit
                        _log_trade(
                            "PARTIAL_EXIT",
                            {
                                "ts": now_dt,
                                "account_id": account_id,
                                "symbol": base,
                                "contract_id": cid,
                                "side": p_side,
                                "qty": int(delta),
                                "entry_price": float(p_entry),
                                "exit_price": float(last_close),
                                "realized_pnl": float(realized),
                                "mode": str(rec.get("mode") or "TREND"),
                                "init_stop": rec.get("init_stop"),
                                "atr_at_entry": rec.get("atr"),
                                "reason": "PARTIAL",
                            },
                        )

            for cid in to_remove:
                open_positions_local.pop(cid, None)

            equity_now = realized_pnl + unrealized_total
            if equity_now > peak_equity:
                peak_equity = equity_now
            dd = peak_equity - equity_now
            if dd > max_dd:
                max_dd = dd

            # Update widget (when enabled)
            if widget is not None:
                widget.update(
                    ts=now_dt,
                    equity=equity_now,
                    realized=realized_pnl,
                    unrealized=unrealized_total,
                    stats={"wins": wins, "losses": losses, "max_dd": max_dd},
                )

            time.sleep(60)
        except KeyboardInterrupt:
            print("Stopped by user.")
            break
        except Exception as e:
            print(f"Loop error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
