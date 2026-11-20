from typing import Any

import pandas as pd
import yfinance as yf

from src.quant.schemas import FinancialStatementInput


class FinancialDataNotFoundError(ValueError):
    """Raised when no financial statement data can be retrieved for a ticker."""


def _extract_latest_value(df: pd.DataFrame, field_names: list[str]) -> float | None:
    if df is None or df.empty:
        return None
    latest_col = df.columns[0]
    for name in field_names:
        if name in df.index:
            val = df.loc[name, latest_col]
            if pd.notna(val):
                return float(val)
    return None


def _extract_fiscal_year(df: pd.DataFrame) -> int | str:
    if df is None or df.empty:
        return "N/A"
    latest_col = df.columns[0]
    if hasattr(latest_col, "year"):
        return int(latest_col.year)
    return str(latest_col)


class YFinanceDataFetcher:
    def fetch_financial_input(self, ticker: str) -> FinancialStatementInput:
        clean_ticker = ticker.strip().upper()
        yf_ticker = yf.Ticker(clean_ticker)

        income_stmt = yf_ticker.income_stmt
        balance_sheet = yf_ticker.balance_sheet
        cashflow = yf_ticker.cashflow
        info: dict[str, Any] = yf_ticker.info or {}

        if (
            (income_stmt is None or income_stmt.empty)
            and (balance_sheet is None or balance_sheet.empty)
            and not info
        ):
            raise FinancialDataNotFoundError(f"No financial data found for ticker: {clean_ticker}")

        fiscal_year = _extract_fiscal_year(income_stmt)
        if fiscal_year == "N/A":
            fiscal_year = _extract_fiscal_year(balance_sheet)

        share_price = info.get("currentPrice") or info.get("regularMarketPrice")
        shares_outstanding = info.get("sharesOutstanding")
        currency = info.get("financialCurrency") or info.get("currency") or "USD"

        return FinancialStatementInput(
            ticker=clean_ticker,
            fiscal_year=fiscal_year,
            currency=str(currency),
            revenue=_extract_latest_value(income_stmt, ["Total Revenue", "Operating Revenue"]),
            cost_of_revenue=_extract_latest_value(income_stmt, ["Cost Of Revenue"]),
            gross_profit=_extract_latest_value(income_stmt, ["Gross Profit"]),
            operating_income=_extract_latest_value(
                income_stmt, ["Operating Income", "Total Operating Income As Reported"]
            ),
            ebitda=_extract_latest_value(income_stmt, ["EBITDA", "Normalized EBITDA"]),
            net_income=_extract_latest_value(
                income_stmt, ["Net Income", "Net Income Common Stockholders"]
            ),
            total_assets=_extract_latest_value(balance_sheet, ["Total Assets"]),
            current_assets=_extract_latest_value(balance_sheet, ["Current Assets"]),
            inventory=_extract_latest_value(balance_sheet, ["Inventory"]),
            cash_and_equivalents=_extract_latest_value(
                balance_sheet,
                ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
            ),
            total_liabilities=_extract_latest_value(
                balance_sheet, ["Total Liabilities Net Minority Interest", "Total Liabilities"]
            ),
            current_liabilities=_extract_latest_value(balance_sheet, ["Current Liabilities"]),
            total_debt=_extract_latest_value(balance_sheet, ["Total Debt"]),
            total_equity=_extract_latest_value(
                balance_sheet, ["Stockholders Equity", "Total Equity Gross Minority Interest"]
            ),
            operating_cash_flow=_extract_latest_value(
                cashflow, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]
            ),
            capital_expenditure=_extract_latest_value(cashflow, ["Capital Expenditure"]),
            share_price=float(share_price) if share_price is not None else None,
            shares_outstanding=float(shares_outstanding)
            if shares_outstanding is not None
            else None,
        )
