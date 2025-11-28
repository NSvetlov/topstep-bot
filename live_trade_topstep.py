import os
import time
from typing import Optional, List, Dict, Tuple
from datetime import datetime

import numpy as np

from topstep_client import TopstepClient
from mnq_backtest import (
    load_mnq_data,
    classify_regime,
    in_entry_session,
    # sizing params
    FIXED_QTY,
    MIN_QTY,
    MAX_CONTRACTS,
    MULTIPLIER,
    RISK_PER_TRADE,
)
from pnl_widget import PnLWidget


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
    atr = row.get("atr", np.nan)
    if FIXED_QTY is not None and FIXED_QTY > 0:
        return min(int(FIXED_QTY), int(MAX_CONTRACTS))

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

    # Track open positions we initiated: contractId -> (side, qty, entry_price)
    open_positions_local: Dict[str, Tuple[str, int, float]] = {}

    # PnL accounting and UI widget
    realized_pnl: float = 0.0
    peak_equity: float = 0.0
    max_dd: float = 0.0
    wins = 0
    losses = 0
    acct_label = os.environ.get("TOPSTEP_ACCOUNT_LABEL") or os.environ.get("TOPSTEP_ACCOUNT_ID") or ""
    title = f"Topstep PnL{(' - ' + acct_label) if acct_label else ''}"
    widget = PnLWidget(title=title)

    # Brackets default enabled with 10/20 ticks; env overrides allowed
    use_bracket_env = os.environ.get("TOPSTEP_USE_BRACKET", "true").strip().lower()
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
                                    open_positions_local[contract_id] = (
                                        side,
                                        int(qty),
                                        float(row["Close"]),
                                    )
                            except Exception as e2:
                                print(f"{ts}: retry without brackets failed: {e2}")
                        else:
                            print(f"{ts}: order response indicates failure -> {resp}")
                    else:
                        print(f"{ts}: submitted {side} {qty} {base} (acct {account_id}, contract {contract_id}) -> {resp}")
                        # Track local position on assumed fill
                        if contract_id not in open_positions_local:
                            open_positions_local[contract_id] = (
                                side,
                                int(qty),
                                float(row["Close"]),
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

            # Compute unrealized PnL across all locally tracked positions
            unrealized_total = 0.0
            to_remove: List[str] = []
            for cid, (p_side, p_qty, p_entry) in list(open_positions_local.items()):
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
                        open_positions_local[cid] = (p_side, qty_live, p_entry)

            for cid in to_remove:
                open_positions_local.pop(cid, None)

            equity_now = realized_pnl + unrealized_total
            if equity_now > peak_equity:
                peak_equity = equity_now
            dd = peak_equity - equity_now
            if dd > max_dd:
                max_dd = dd

            # Update widget
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
