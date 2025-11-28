import os
from typing import Any, Dict, List, Optional

import requests


class TopstepClient:
    """
    Minimal REST client for Topstep endpoints provided.

    Auth
    - Use header 'X-API-Key: <API_KEY>' on every request.

    Endpoints (relative to base URL)
    - POST /api/Account/search     {"onlyActiveAccounts": true}
    - POST /api/Contract/available {"live": true}
    - POST /api/Order/place        {accountId, contractId, type, side, size, ...}

    Auth flow (TopstepX loginKey)
    1) POST /api/Auth/loginKey with { userName, apiKey } to get JWT token
    2) Use Authorization: Bearer <token> for all subsequent requests

    Env vars
    - TOPSTEP_BASE_URL (e.g., https://api.topstepx.com)
    - TOPSTEP_USERNAME (TopstepX login username)
    - TOPSTEP_API_KEY (optional; otherwise read from key file)
    - TOPSTEP_ACCOUNT_ID (optional; will search if missing)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        username: Optional[str] = None,
        account_id: Optional[str] = None,
        header_name: str = "X-API-Key",
        timeout: float = 10.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.username = username
        self.account_id = account_id
        self.header_name = header_name
        self.timeout = timeout
        # fixed endpoints per spec
        self._ep_accounts_search = "/api/Account/search"
        self._ep_contracts_available = "/api/Contract/available"
        self._ep_order_place = "/api/Order/place"
        self._ep_positions_open = "/api/Position/searchOpen"
        self._ep_auth_login_key = "/api/Auth/loginKey"
        self._session = session or requests.Session()
        self._debug = os.environ.get("TOPSTEP_DEBUG", "").strip().lower() in ("1", "true", "yes")
        self._token: Optional[str] = None

    @classmethod
    def from_env(cls) -> "TopstepClient":
        base_url = os.environ.get("TOPSTEP_BASE_URL", "https://api.topstepx.com").strip() or "https://api.topstepx.com"
        api_key = os.environ.get("TOPSTEP_API_KEY", "").strip()
        username = os.environ.get("TOPSTEP_USERNAME", "").strip() or None
        # Fallback: read API key from local file if env var not set
        if not api_key:
            for fname in ("topstep_api_key.txt", ".topstep_api_key", "TOPSTEP_API_KEY.txt"):
                if os.path.exists(fname):
                    try:
                        with open(fname, "r", encoding="utf-8") as f:
                            api_key = f.read().strip()
                            break
                    except Exception:
                        pass
        account_id = os.environ.get("TOPSTEP_ACCOUNT_ID", "").strip() or None
        if not base_url or not api_key or not username:
            raise RuntimeError(
                "Missing TOPSTEP_BASE_URL, TOPSTEP_USERNAME, or API key (env or key file)."
            )
        header_name = os.environ.get("TOPSTEP_HEADER_NAME", "X-API-Key").strip()
        client = cls(
            base_url=base_url,
            api_key=api_key,
            username=username,
            account_id=account_id,
            header_name=header_name,
        )
        # Perform login to obtain JWT
        client.login_key()
        return client

    def _headers(self) -> Dict[str, str]:
        if not self._token:
            # attempt lazy login
            self.login_key()
        if not self._token:
            raise RuntimeError("No auth token available; loginKey failed.")
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    # --- Auth ---
    def login_key(self) -> None:
        """Authenticate using loginKey to obtain JWT token."""
        if not self.username or not self.api_key:
            raise RuntimeError("Missing username or API key for loginKey.")
        url = self._url(self._ep_auth_login_key)
        payload = {"userName": self.username, "apiKey": self.api_key}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        r = self._session.post(url, json=payload, headers=headers, timeout=self.timeout)
        if self._debug and r.status_code != 200:
            print(f"DEBUG login_key {url} -> {r.status_code}: {r.text[:500]}")
        try:
            data = r.json()
        except Exception:
            data = None
        if r.status_code != 200:
            raise RuntimeError(f"Auth failed (HTTP {r.status_code}). Check TOPSTEP_USERNAME/API key.")
        # Parse token and success flag robustly
        token = None
        success = None
        if isinstance(data, dict):
            token = data.get("token") or data.get("accessToken") or data.get("jwt")
            success = data.get("success")
            # sometimes wrapped under 'data'
            if not token and isinstance(data.get("data"), dict):
                dd = data["data"]
                token = dd.get("token") or dd.get("accessToken") or dd.get("jwt")
                if success is None:
                    success = dd.get("success")
        if success is False or not token:
            raise RuntimeError("Auth failed: success=false or token missing. Check TOPSTEP_USERNAME/API key.")
        self._token = str(token)

    # --- Accounts ---
    def search_accounts(self, only_active: bool = True) -> Any:
        payload = {"onlyActiveAccounts": bool(only_active)}
        url = self._url(self._ep_accounts_search)
        r = self._session.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
        if self._debug and r.status_code != 200:
            print(f"DEBUG search_accounts {url} -> {r.status_code}: {r.text[:500]}")
        r.raise_for_status()
        return r.json()

    # --- Contracts ---
    def available_contracts(self, live: bool = True) -> Any:
        payload = {"live": bool(live)}
        url = self._url(self._ep_contracts_available)
        r = self._session.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
        if self._debug:
            print(f"DEBUG available_contracts {url} -> {r.status_code}: {r.text[:500]}")
        r.raise_for_status()
        return r.json()

    # --- Orders ---
    def place_market_order(
        self,
        account_id: str,
        contract_id: str,
        side: str,
        size: int,
        stop_ticks: Optional[int] = None,
        take_ticks: Optional[int] = None,
    ) -> Any:
        # Map side to API enum (0 = BUY, 1 = SELL as per provided example)
        side_map = {"BUY": 0, "SELL": 1, "LONG": 0, "SHORT": 1}
        side_val = side_map.get(side.upper())
        if side_val is None:
            raise ValueError("side must be one of BUY/SELL/LONG/SHORT")

        payload: Dict[str, Any] = {
            "accountId": account_id,
            "contractId": contract_id,
            "type": 2,  # Market
            "side": side_val,
            "size": int(size),
        }
        if stop_ticks is not None:
            payload["stopLossBracket"] = {"ticks": int(stop_ticks), "type": 1}
        if take_ticks is not None:
            payload["takeProfitBracket"] = {"ticks": int(take_ticks), "type": 1}

        url = self._url(self._ep_order_place)
        r = self._session.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
        if self._debug and r.status_code != 200:
            print(f"DEBUG place_market_order {url} -> {r.status_code}: {r.text[:500]}")
        r.raise_for_status()
        return r.json()

    # --- Positions ---
    def search_open_positions(self, account_id: str) -> Any:
        payload = {"accountId": account_id}
        url = self._url(self._ep_positions_open)
        r = self._session.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
        if self._debug and r.status_code != 200:
            print(f"DEBUG search_open_positions {url} -> {r.status_code}: {r.text[:500]}")
        r.raise_for_status()
        return r.json()

    # --- Helpers ---
    def resolve_account_id(self) -> str:
        if self.account_id:
            return self.account_id
        data = self.search_accounts(only_active=True)
        # Try common shapes: list or {data: [...]} or {accounts: [...]}
        candidates: List[Dict[str, Any]]
        if isinstance(data, list):
            candidates = data
        else:
            candidates = data.get("accounts") or data.get("data") or []  # type: ignore[attr-defined]
        if not candidates:
            raise RuntimeError("No accounts returned from /api/Account/search")
        # Pick the first one by default
        acct_id = str(candidates[0].get("id") or candidates[0].get("accountId") or candidates[0].get("accountID"))
        if not acct_id:
            raise RuntimeError("Could not parse accountId from /api/Account/search response")
        self.account_id = acct_id
        return acct_id

    def _extract_items_list(self, data: Any) -> List[Dict[str, Any]]:
        # Try multiple shapes to extract a list of contract dicts
        if isinstance(data, list):
            return data  # already a list
        if not isinstance(data, dict):
            return []
        # common keys that may contain list or nested object
        for key in ["contracts", "data", "items", "result", "payload", "response"]:
            if key in data:
                val = data[key]
                if isinstance(val, list):
                    return val
                if isinstance(val, dict):
                    nested = self._extract_items_list(val)
                    if nested:
                        return nested
        # if there is only one contract under some key
        return []

    def resolve_contract_id(self, symbol: str, live: bool = True) -> str:
        data = self.available_contracts(live=live)
        items: List[Dict[str, Any]] = self._extract_items_list(data)
        if not items and live:
            # Fallback: some practice environments expose contracts under live=false
            if self._debug:
                print("DEBUG resolve_contract_id: no live contracts; retrying with live=false")
            data2 = self.available_contracts(live=False)
            items = self._extract_items_list(data2)
        if not items:
            raise RuntimeError("No contracts returned from /api/Contract/available")

        sym_upper = symbol.upper()
        def get_fields(d: Dict[str, Any]) -> List[str]:
            keys = [
                "symbol", "name", "code", "displayName", "productSymbol",
                "contract", "contractName", "exchangeSymbol",
            ]
            vals: List[str] = []
            for k in keys:
                v = d.get(k)
                if isinstance(v, str):
                    vals.append(v)
            return vals

        for it in items:
            vals = get_fields(it)
            if any(v.upper() == sym_upper for v in vals):
                cid = it.get("id") or it.get("contractId") or it.get("contractID")
                if cid is not None:
                    return str(cid)
        # fallback: startswith match
        for it in items:
            vals = get_fields(it)
            if any(v.upper().startswith(sym_upper) for v in vals):
                cid = it.get("id") or it.get("contractId") or it.get("contractID")
                if cid is not None:
                    return str(cid)
        raise RuntimeError(f"Could not find contractId for symbol '{symbol}'. Provide TOPSTEP_CONTRACT_ID.")
