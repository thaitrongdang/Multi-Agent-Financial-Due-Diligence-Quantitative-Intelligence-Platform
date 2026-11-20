import json
import os
from collections.abc import Generator
from typing import Any

import httpx

DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

DEMO_PRESETS: dict[str, dict[str, Any]] = {
    "FPT.VN (Healthy Growth & Bond Coverage)": {
        "ticker": "FPT.VN",
        "fiscal_year": 2025,
        "query": "Assess liquidity ratio, short-term bond maturity, and cash coverage.",
        "financial_input": {
            "ticker": "FPT.VN",
            "fiscal_year": 2025,
            "currency": "VND",
            "revenue": 62800.0,
            "cost_of_revenue": 38900.0,
            "operating_income": 11200.0,
            "ebitda": 13500.0,
            "net_income": 9400.0,
            "total_assets": 68000.0,
            "current_assets": 36000.0,
            "inventory": 2400.0,
            "cash_and_equivalents": 14500.0,
            "current_liabilities": 24000.0,
            "total_debt": 15000.0,
            "total_equity": 32000.0,
            "operating_cash_flow": 11800.0,
            "capital_expenditure": -3200.0,
            "share_price": 138.0,
            "shares_outstanding": 1460.0,
        },
        "news_headlines": [
            "FPT records 22% software export growth in Japan and APAC",
            "FPT expands AI factory infrastructure investment",
        ],
        "filing_markdown": (
            "| Debt Facility | Principal (VND Bn) | Maturity |\n"
            "| --- | --- | --- |\n"
            "| Corporate Bond FPTB2426 | 1250 | Q3 2026 |\n"
            "| Working Capital Credit Line | 4800 | Q4 2026 |\n\n"
            "Footnote 18: Corporate Bond FPTB2426 carries a fixed coupon of 6.8% "
            "and is fully covered by cash reserves."
        ),
    },
    "HPG.VN (Heavy Capex Expansion)": {
        "ticker": "HPG.VN",
        "fiscal_year": 2025,
        "query": "Evaluate leverage, net debt to EBITDA, and Dung Quat 2 capex commitments.",
        "financial_input": {
            "ticker": "HPG.VN",
            "fiscal_year": 2025,
            "currency": "VND",
            "revenue": 138000.0,
            "cost_of_revenue": 118000.0,
            "operating_income": 16500.0,
            "ebitda": 24000.0,
            "net_income": 12200.0,
            "total_assets": 210000.0,
            "current_assets": 82000.0,
            "inventory": 39000.0,
            "cash_and_equivalents": 22000.0,
            "current_liabilities": 68000.0,
            "total_debt": 80000.0,
            "total_equity": 110000.0,
            "operating_cash_flow": 18500.0,
            "capital_expenditure": -24500.0,
            "share_price": 28.5,
            "shares_outstanding": 6396.0,
        },
        "news_headlines": [
            "HPG hot-rolled coil output reaches new quarterly peak",
            "Steel margin recovery supports HPG earnings expansion",
        ],
        "filing_markdown": (
            "| Project Commitment | Committed Capex (VND Bn) | Status |\n"
            "| --- | --- | --- |\n"
            "| Dung Quat 2 Integrated Complex | 24500 | Phase 1 Commissioning |\n"
            "| Container Manufacturing Plant | 1800 | Operational |"
        ),
    },
    "RISK.VN (Distressed Red-Flag Case)": {
        "ticker": "RISK.VN",
        "fiscal_year": 2025,
        "query": "Identify liquidity red flags, covenant breach risks, and cash flow deficits.",
        "financial_input": {
            "ticker": "RISK.VN",
            "fiscal_year": 2025,
            "currency": "VND",
            "revenue": 42000.0,
            "cost_of_revenue": 35000.0,
            "operating_income": 2800.0,
            "ebitda": 4200.0,
            "net_income": 1200.0,
            "total_assets": 95000.0,
            "current_assets": 30000.0,
            "inventory": 14000.0,
            "cash_and_equivalents": 3000.0,
            "current_liabilities": 40000.0,
            "total_debt": 50000.0,
            "total_equity": 20000.0,
            "operating_cash_flow": -3500.0,
            "capital_expenditure": -2100.0,
            "share_price": 11.2,
            "shares_outstanding": 850.0,
        },
        "news_headlines": [
            "RISK.VN faces credit rating downgrade amid working capital strain",
            "Bondholders request additional collateral following cash flow decline",
        ],
        "filing_markdown": (
            "Going Concern Note: The company obtained a temporary covenant waiver on its "
            "3100 VND Bn syndicated loan after breaching minimum interest coverage.\n\n"
            "| Overdue Facility | Balance (VND Bn) | Waiver Expiry |\n"
            "| --- | --- | --- |\n"
            "| Syndicated Term Loan | 3100 | June 2026 |"
        ),
    },
}


def parse_sse_line(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped.startswith("data: "):
        return None
    raw_json = stripped.removeprefix("data: ")
    return json.loads(raw_json)


def check_backend_health(api_base_url: str = DEFAULT_API_URL) -> bool:
    try:
        res = httpx.get(f"{api_base_url.rstrip('/')}/api/v1/health", timeout=3.0)
        return res.status_code == 200
    except httpx.HTTPError:
        return False


def ingest_text_document(
    api_base_url: str,
    ticker: str,
    fiscal_year: int | str,
    markdown_content: str,
    page_number: int = 1,
) -> dict[str, Any]:
    res = httpx.post(
        f"{api_base_url.rstrip('/')}/api/v1/documents/ingest-text",
        json={
            "ticker": ticker,
            "fiscal_year": fiscal_year,
            "page_number": page_number,
            "markdown_content": markdown_content,
        },
        timeout=30.0,
    )
    res.raise_for_status()
    return res.json()


def ingest_pdf_document(
    api_base_url: str,
    ticker: str,
    fiscal_year: int | str,
    filename: str,
    file_bytes: bytes,
) -> dict[str, Any]:
    res = httpx.post(
        f"{api_base_url.rstrip('/')}/api/v1/documents/ingest-pdf",
        data={"ticker": ticker, "fiscal_year": str(fiscal_year)},
        files={"file": (filename, file_bytes, "application/pdf")},
        timeout=60.0,
    )
    res.raise_for_status()
    return res.json()


def stream_due_diligence(
    api_base_url: str,
    payload: dict[str, Any],
) -> Generator[dict[str, Any], None, None]:
    url = f"{api_base_url.rstrip('/')}/api/v1/analyze/stream"
    with httpx.stream("POST", url, json=payload, timeout=120.0) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            event = parse_sse_line(line)
            if event is not None:
                yield event
