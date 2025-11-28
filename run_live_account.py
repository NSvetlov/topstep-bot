import os
import sys
from typing import Any, Dict, List, Optional

from topstep_client import TopstepClient


def _extract_accounts(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("accounts", "data", "items", "result"):
            v = data.get(k)
            if isinstance(v, list):
                return v
    return []


def _get_id(acct: Dict[str, Any]) -> Optional[str]:
    for k in ("id", "accountId", "accountID"):
        v = acct.get(k)
        if v is not None and str(v):
            return str(v)
    return None


def _get_name(acct: Dict[str, Any]) -> str:
    for k in ("name", "displayName", "accountName", "nickname"):
        v = acct.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # fallback to id
    return _get_id(acct) or "(unknown)"


def choose_account(accounts: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Print list for user
    rows = [(i, _get_id(a) or "", _get_name(a)) for i, a in enumerate(accounts, 1)]
    print("Available accounts:")
    for i, aid, name in rows:
        print(f"  {i:2d}) {name}  [{aid}]")

    raw = input("Enter account name, ID, or number: ").strip()
    if not raw:
        return accounts[0]

    # number selection
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(accounts):
            return accounts[idx - 1]

    # id exact
    for a in accounts:
        if (_get_id(a) or "").lower() == raw.lower():
            return a

    # name exact
    for a in accounts:
        if _get_name(a).lower() == raw.lower():
            return a

    # substring match (first)
    for a in accounts:
        if raw.lower() in _get_name(a).lower():
            return a

    print("No match; defaulting to first account.")
    return accounts[0]


def main() -> None:
    # Ensure core auth is available
    if not os.environ.get("TOPSTEP_USERNAME"):
        print("Missing TOPSTEP_USERNAME. Set it in your shell.")
        sys.exit(1)
    if not (os.environ.get("TOPSTEP_API_KEY") or any(os.path.exists(p) for p in ("topstep_api_key.txt", ".topstep_api_key", "TOPSTEP_API_KEY.txt"))):
        print("Missing TOPSTEP_API_KEY (env or key file). Set it before running.")
        sys.exit(1)

    try:
        client = TopstepClient.from_env()
    except Exception as e:
        print(f"Auth failed: {e}")
        sys.exit(1)

    try:
        accts_raw = client.search_accounts(only_active=True)
        accounts = _extract_accounts(accts_raw)
    except Exception as e:
        print(f"Could not fetch accounts: {e}")
        sys.exit(1)

    if not accounts:
        print("No accounts found for this login.")
        sys.exit(1)

    chosen = choose_account(accounts)
    acct_id = _get_id(chosen)
    if not acct_id:
        print("Selected account has no ID; cannot continue.")
        sys.exit(1)
    acct_name = _get_name(chosen)

    # Set env for the live trader
    os.environ["TOPSTEP_ACCOUNT_ID"] = acct_id
    os.environ["TOPSTEP_ACCOUNT_LABEL"] = acct_name

    # Import and run live trader in-process
    from live_trade_topstep import main as run_live

    print(f"Running live trader for account: {acct_name} [{acct_id}]\n")
    run_live()


if __name__ == "__main__":
    main()

