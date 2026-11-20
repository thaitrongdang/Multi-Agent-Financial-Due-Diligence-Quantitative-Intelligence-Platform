from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.quant.data_fetcher import FinancialDataNotFoundError, YFinanceDataFetcher
from src.quant.ratio_calculator import calculate_financial_metrics
from src.quant.schemas import FinancialStatementInput


def test_calculate_financial_metrics_standard_case() -> None:
    sample = FinancialStatementInput(
        ticker="aapl",
        fiscal_year=2025,
        currency="USD",
        revenue=400_000.0,
        cost_of_revenue=220_000.0,
        operating_income=120_000.0,
        ebitda=135_000.0,
        net_income=100_000.0,
        total_assets=350_000.0,
        current_assets=150_000.0,
        inventory=10_000.0,
        cash_and_equivalents=30_000.0,
        current_liabilities=100_000.0,
        total_debt=110_000.0,
        total_equity=60_000.0,
        operating_cash_flow=115_000.0,
        capital_expenditure=-15_000.0,
        share_price=200.0,
        shares_outstanding=10_000.0,
    )

    metrics = calculate_financial_metrics(sample)

    assert metrics.ticker == "AAPL"
    assert metrics.current_ratio == 1.5
    assert metrics.quick_ratio == 1.4
    assert metrics.gross_margin == 0.45
    assert metrics.operating_margin == 0.3
    assert metrics.net_profit_margin == 0.25
    assert metrics.roe == round(100_000.0 / 60_000.0, 4)
    assert metrics.roa == round(100_000.0 / 350_000.0, 4)
    assert metrics.debt_to_equity == round(110_000.0 / 60_000.0, 4)
    assert metrics.net_debt == 80_000.0
    assert metrics.net_debt_to_ebitda == round(80_000.0 / 135_000.0, 4)
    assert metrics.free_cash_flow == 100_000.0
    assert metrics.eps == 10.0
    assert metrics.pe_ratio == 20.0
    assert metrics.red_flags == []


def test_calculate_financial_metrics_zero_division_and_negative_earnings() -> None:
    edge_case = FinancialStatementInput(
        ticker="RISK",
        fiscal_year=2025,
        revenue=50_000.0,
        net_income=-5_000.0,
        ebitda=-2_000.0,
        current_assets=10_000.0,
        current_liabilities=0.0,
        total_debt=50_000.0,
        total_equity=0.0,
        share_price=25.0,
        shares_outstanding=1_000.0,
    )

    metrics = calculate_financial_metrics(edge_case)

    assert metrics.current_ratio is None
    assert metrics.quick_ratio is None
    assert metrics.roe is None
    assert metrics.debt_to_equity is None
    assert metrics.net_debt_to_ebitda is None
    assert metrics.eps == -5.0
    assert metrics.pe_ratio is None


def test_detect_red_flags_triggers_warnings() -> None:
    distressed = FinancialStatementInput(
        ticker="DEBT",
        fiscal_year=2025,
        net_income=10_000.0,
        operating_cash_flow=-4_000.0,
        current_assets=80_000.0,
        current_liabilities=100_000.0,
        total_debt=250_000.0,
        cash_and_equivalents=10_000.0,
        total_equity=100_000.0,
        ebitda=50_000.0,
    )

    metrics = calculate_financial_metrics(distressed)

    assert len(metrics.red_flags) == 4
    assert any("Liquidity risk" in f for f in metrics.red_flags)
    assert any("High leverage" in f for f in metrics.red_flags)
    assert any("Debt service strain" in f for f in metrics.red_flags)
    assert any("Earnings quality warning" in f for f in metrics.red_flags)


@patch("src.quant.data_fetcher.yf.Ticker")
def test_yfinance_fetcher_parses_statements(mock_ticker_cls: MagicMock) -> None:
    col_date = pd.Timestamp("2025-12-31")
    mock_instance = MagicMock()
    mock_instance.income_stmt = pd.DataFrame(
        {col_date: [500_000.0, 200_000.0, 80_000.0]},
        index=["Total Revenue", "Gross Profit", "Net Income"],
    )
    mock_instance.balance_sheet = pd.DataFrame(
        {col_date: [400_000.0, 120_000.0, 80_000.0, 150_000.0]},
        index=["Total Assets", "Current Assets", "Current Liabilities", "Stockholders Equity"],
    )
    mock_instance.cashflow = pd.DataFrame(
        {col_date: [95_000.0, -15_000.0]},
        index=["Operating Cash Flow", "Capital Expenditure"],
    )
    mock_instance.info = {
        "currentPrice": 150.0,
        "sharesOutstanding": 10_000.0,
        "financialCurrency": "USD",
    }
    mock_ticker_cls.return_value = mock_instance

    fetcher = YFinanceDataFetcher()
    parsed = fetcher.fetch_financial_input("FPT.VN")
    metrics = calculate_financial_metrics(parsed)

    assert parsed.ticker == "FPT.VN"
    assert parsed.fiscal_year == 2025
    assert metrics.current_ratio == 1.5
    assert metrics.free_cash_flow == 80_000.0


@patch("src.quant.data_fetcher.yf.Ticker")
def test_yfinance_fetcher_raises_on_invalid_ticker(mock_ticker_cls: MagicMock) -> None:
    mock_instance = MagicMock()
    mock_instance.income_stmt = pd.DataFrame()
    mock_instance.balance_sheet = pd.DataFrame()
    mock_instance.cashflow = pd.DataFrame()
    mock_instance.info = {}
    mock_ticker_cls.return_value = mock_instance

    fetcher = YFinanceDataFetcher()
    with pytest.raises(FinancialDataNotFoundError):
        fetcher.fetch_financial_input("INVALID_TICKER_XYZ")
