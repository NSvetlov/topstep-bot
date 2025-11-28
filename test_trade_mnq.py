import os
import time
from typing import Optional

from topstep_client import TopstepClient


def main():
    # Config via env vars (with safe defaults)
    username = os.environ.get("TOPSTEP_USERNAME", "").strip()
    # API key is read from env TOPSTEP_API_KEY or key file in TopstepClient.from_env()
    base_symbol = os.environ.get("TOPSTEP_TEST_SYMBOL", "MNQ").strip()  # symbol used to resolve contract
    explicit_contract_id = os.environ.get("TOPSTEP_CONTRACT_ID", "").strip() or None
    qty_env = os.environ.get("TOPSTEP_TEST_QTY", "1").strip()
    hold_secs_env = os.environ.get("TOPSTEP_TEST_HOLD_SECONDS", "3600").strip()
    side_env = os.environ.get("TOPSTEP_TEST_SIDE", "BUY").strip().upper()
    live_flag = os.environ.get("TOPSTEP_CONTRACT_LIVE", "false").strip().lower() in ("1", "true", "yes")

    if not username:
        raise RuntimeError("Set TOPSTEP_USERNAME to your TopstepX login username.")

    try:
        qty = max(1, int(qty_env))
    except Exception:
        qty = 1
    try:
        hold_secs = max(60, int(hold_secs_env))
    except Exception:
        hold_secs = 3600

    if side_env not in ("BUY", "SELL"):
        side_env = "BUY"

    # Create client (performs loginKey and caches JWT)
    client = TopstepClient.from_env()

    # Resolve account
    account_id = os.environ.get("TOPSTEP_ACCOUNT_ID", "").strip() or client.resolve_account_id()

    # Resolve contract
    contract_id: Optional[str] = explicit_contract_id
    if not contract_id:
        contract_id = client.resolve_contract_id(symbol=base_symbol, live=live_flag)

    # Place entry market order (no brackets for test to avoid broker setting conflicts)
    resp = client.place_market_order(
        account_id=account_id,
        contract_id=contract_id,
        side=side_env,
        size=qty,
        stop_ticks=None,
        take_ticks=None,
    )
    print(f"Submitted {side_env} {qty} {base_symbol} (acct {account_id}, contract {contract_id}) -> {resp}")

    # Hold for the requested time, then send the opposite side to flatten
    print(f"Sleeping {hold_secs} seconds before exit...")
    try:
        time.sleep(hold_secs)
    except KeyboardInterrupt:
        print("Interrupted before exit; not placing the closing order.")
        return

    exit_side = "SELL" if side_env == "BUY" else "BUY"
    resp2 = client.place_market_order(
        account_id=account_id,
        contract_id=contract_id,
        side=exit_side,
        size=qty,
        stop_ticks=None,
        take_ticks=None,
    )
    print(f"Submitted {exit_side} {qty} {base_symbol} (acct {account_id}, contract {contract_id}) -> {resp2}")


if __name__ == "__main__":
    main()

