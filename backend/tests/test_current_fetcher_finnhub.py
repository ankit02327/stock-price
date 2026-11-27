from unittest.mock import MagicMock

import pytest
import requests

from backend.data_fetching.current_fetcher import CurrentFetcher


def _make_response(status_code=200, json_data=None, http_error=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    if http_error is not None:
        resp.raise_for_status.side_effect = http_error
    else:
        resp.raise_for_status.return_value = None
    return resp


def test_fetch_from_finnhub_success_parses_quote_and_profile(monkeypatch):
    fetcher = CurrentFetcher()
    fetcher.finnhub_api_key = "test-finnhub-key"

    requested_urls = []

    def fake_get(url, timeout=10):
        requested_urls.append((url, timeout))
        if "/quote?" in url:
            return _make_response(200, {"c": 192.45})
        if "/stock/profile2?" in url:
            return _make_response(200, {"name": "Apple Inc."})
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(requests, "get", fake_get)

    price, company_name, hist_date = fetcher._fetch_from_finnhub(" aapl ")
    assert price == 192.45
    assert company_name == "Apple Inc."
    assert hist_date is None
    assert requested_urls[0][0] == "https://finnhub.io/api/v1/quote?symbol=AAPL&token=test-finnhub-key"
    assert requested_urls[1][0] == "https://finnhub.io/api/v1/stock/profile2?symbol=AAPL&token=test-finnhub-key"


def test_fetch_from_finnhub_falls_back_to_symbol_when_profile_fails(monkeypatch):
    fetcher = CurrentFetcher()
    fetcher.finnhub_api_key = "test-finnhub-key"

    def fake_get(url, timeout=10):
        if "/quote?" in url:
            return _make_response(200, {"c": 50.0})
        return _make_response(500, {})

    monkeypatch.setattr(requests, "get", fake_get)

    price, company_name, hist_date = fetcher._fetch_from_finnhub("MSFT")
    assert price == 50.0
    assert company_name == "MSFT"
    assert hist_date is None


@pytest.mark.parametrize("status_code", [401, 429])
def test_fetch_from_finnhub_raises_on_http_errors(monkeypatch, status_code):
    fetcher = CurrentFetcher()
    fetcher.finnhub_api_key = "test-finnhub-key"

    err = requests.exceptions.HTTPError(f"{status_code} Error")

    def fake_get(url, timeout=10):
        return _make_response(status_code=status_code, http_error=err)

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(requests.exceptions.HTTPError):
        fetcher._fetch_from_finnhub("AAPL")


def test_fetch_from_finnhub_raises_when_price_missing_or_api_key_unset(monkeypatch):
    fetcher = CurrentFetcher()
    fetcher.finnhub_api_key = ""

    with pytest.raises(ValueError, match="Finnhub API key not configured"):
        fetcher._fetch_from_finnhub("AAPL")

    fetcher.finnhub_api_key = "test-finnhub-key"
    monkeypatch.setattr(requests, "get", lambda url, timeout=10: _make_response(200, {"c": None}))

    with pytest.raises(ValueError, match="No price data available"):
        fetcher._fetch_from_finnhub("AAPL")
