"""Shared HTTP session with retry logic for all external requests."""

from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session = None


def get_session():
    """Return a requests.Session with automatic retries on transient failures."""
    global _session
    if _session is None:
        _session = Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        _session.mount("https://", adapter)
        _session.mount("http://", adapter)
        _session.headers["User-Agent"] = "Mozilla/5.0 (compatible; MarketFlowsBot/1.0)"
    return _session
