from unittest.mock import MagicMock, patch

import httpx

from src.ui.client import DEMO_PRESETS, check_backend_health, parse_sse_line


def test_parse_sse_line_extracts_valid_json_event() -> None:
    line = 'data: {"event": "node_update", "node": "quant", "trace": ["Computed ROE"]}'
    parsed = parse_sse_line(line)
    assert parsed is not None
    assert parsed["event"] == "node_update"
    assert parsed["node"] == "quant"


def test_parse_sse_line_ignores_non_data_lines() -> None:
    assert parse_sse_line("") is None
    assert parse_sse_line(": keep-alive") is None


def test_demo_presets_contain_complete_structures() -> None:
    assert len(DEMO_PRESETS) >= 3
    for preset in DEMO_PRESETS.values():
        assert "ticker" in preset
        assert "financial_input" in preset
        assert "filing_markdown" in preset


@patch("src.ui.client.httpx.get")
def test_check_backend_health_handles_online_and_offline(mock_get: MagicMock) -> None:
    mock_ok = MagicMock(status_code=200)
    mock_get.return_value = mock_ok
    assert check_backend_health("http://localhost:8000") is True

    mock_get.side_effect = httpx.ConnectError("Connection refused")
    assert check_backend_health("http://localhost:8000") is False
