import os
import sys
import json
from topstep_client import TopstepClient


def main():
    try:
        c = TopstepClient.from_env()
    except Exception as e:
        print(f"Auth init failed: {e}")
        sys.exit(1)

    print(f"Base URL: {c.base_url}")
    try:
        a = c.search_accounts(True)
        print("Accounts OK:\n", json.dumps(a, indent=2)[:800])
    except Exception as e:
        print(f"Account search failed: {e}")
        return

    try:
        con = c.available_contracts(True)
        print("Contracts OK:\n", json.dumps(con, indent=2)[:800])
    except Exception as e:
        print(f"Contracts failed: {e}")
        return

    try:
        acct_id = c.resolve_account_id()
        pos = c.search_open_positions(acct_id)
        print("Open positions OK:\n", json.dumps(pos, indent=2)[:800])
    except Exception as e:
        print(f"Positions failed: {e}")


if __name__ == "__main__":
    main()
