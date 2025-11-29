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


def compute_entry_signal(row) -> Optional[str]:
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

    ATR_STOP_MULT = _env_float("TOPSTEP_ATR_STOP_MULT", 2.0)
    BE_TRIGGER_ATR = _env_float("TOPSTEP_BE_TRIGGER_ATR", 1.0)
    BE_OFFSET_ATR = _env_float("TOPSTEP_BE_OFFSET_ATR", 0.25)
    TRAIL1_TRIGGER_ATR = _env_float("TOPSTEP_TRAIL1_TRIGGER_ATR", 2.0)
    TRAIL1_MULT = _env_float("TOPSTEP_TRAIL1_MULT", 1.5)
    TRAIL2_TRIGGER_ATR = _env_float("TOPSTEP_TRAIL2_TRIGGER_ATR", 3.0)
    TRAIL2_MULT = _env_float("TOPSTEP_TRAIL2_MULT", 1.0)
    TIME_STOP_MIN = _env_int("TOPSTEP_TIME_STOP_MIN", 120)
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

    # Main 1-minute polling loop (no entry session restriction)
    while True:
        try:
            ts = None
            last_price: Dict[str, float] = {}
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

                side = compute_entry_signal(row)
                if not side:
                    print(f"{ts}: no entry signal for {base}.")
                    continue

                qty = compute_qty(row)
                if not qty:
                    print(f"{ts}: no valid qty for {base}; skip.")
                    continue

                try:
                    resp = client.place_market_order(
                        account_id=account_id,
                        contract_id=contract_id,
                        side=("BUY" if side == "LONG" else "SELL"),
                        size=qty,
                        stop_ticks=stop_ticks_val,
                        take_ticks=take_ticks_val,
                    )
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
                                        "mfe_price": float(row["Close"]),
                                        "exit_sent": False,
                                    }
                            except Exception as e2:
                                print(f"{ts}: retry without brackets failed: {e2}")
                        else:
                            print(f"{ts}: order response indicates failure -> {resp}")
                    else:
                        print(f"{ts}: submitted {side} {qty} {base} (acct {account_id}, contract {contract_id}) -> {resp}")
                        # Track local position on assumed fill
                        if contract_id not in open_positions_local:
                            open_positions_local[contract_id] = {
                                "side": side,
                                "qty": int(qty),
                                "entry_price": float(row["Close"]),
                                "entry_time": datetime.now(),
                                "atr": float(row.get("atr") or 0.0),
                                "mfe_price": float(row["Close"]),
                                "exit_sent": False,
                            }
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
                mfe_price: float = float(rec.get("mfe_price") or p_entry)
                exit_sent: bool = bool(rec.get("exit_sent", False))

                # Update MFE price
                if p_side == "LONG":
                    mfe_price = max(mfe_price, last_close)
                else:
                    mfe_price = min(mfe_price, last_close)
                rec["mfe_price"] = mfe_price

                # Exit rules (only if position is live and we haven't already sent an exit)
                qty_live = int(active_contracts.get(cid, 0))
                should_exit = False
                if qty_live > 0 and not exit_sent:
                    # Time stop
                    minutes_in_trade = (now_dt - p_time).total_seconds() / 60.0
                    if TIME_STOP_MIN > 0 and minutes_in_trade >= TIME_STOP_MIN:
                        should_exit = True

                    # Force flat window
                    if not should_exit and ENFORCE_FORCE_FLAT and past_force_flat(now_dt):
                        should_exit = True

                    # Price-based stops using entry ATR
                    if not should_exit and atr_entry > 0:
                        # initial stop
                        if p_side == "LONG":
                            stop_price = p_entry - ATR_STOP_MULT * atr_entry
                            move_fav = last_close - p_entry
                            # breakeven
                            if move_fav >= BE_TRIGGER_ATR * atr_entry:
                                stop_price = max(stop_price, p_entry + BE_OFFSET_ATR * atr_entry)
                            # trails
                            if move_fav >= TRAIL1_TRIGGER_ATR * atr_entry:
                                stop_price = max(stop_price, mfe_price - TRAIL1_MULT * atr_entry)
                            if move_fav >= TRAIL2_TRIGGER_ATR * atr_entry:
                                stop_price = max(stop_price, mfe_price - TRAIL2_MULT * atr_entry)
                            if last_close <= stop_price:
                                should_exit = True
                        else:  # SHORT
                            stop_price = p_entry + ATR_STOP_MULT * atr_entry
                            move_fav = p_entry - last_close
                            if move_fav >= BE_TRIGGER_ATR * atr_entry:
                                stop_price = min(stop_price, p_entry - BE_OFFSET_ATR * atr_entry)
                            if move_fav >= TRAIL1_TRIGGER_ATR * atr_entry:
                                stop_price = min(stop_price, mfe_price + TRAIL1_MULT * atr_entry)
                            if move_fav >= TRAIL2_TRIGGER_ATR * atr_entry:
                                stop_price = min(stop_price, mfe_price + TRAIL2_MULT * atr_entry)
                            if last_close >= stop_price:
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
