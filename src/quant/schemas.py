from pydantic import BaseModel, Field


class FinancialStatementInput(BaseModel):
    ticker: str
    fiscal_year: int | str
    currency: str = "USD"

    # Income Statement
    revenue: float | None = None
    cost_of_revenue: float | None = None
    gross_profit: float | None = None
    operating_income: float | None = None
    ebitda: float | None = None
    net_income: float | None = None

    # Balance Sheet
    total_assets: float | None = None
    current_assets: float | None = None
    inventory: float | None = None
    cash_and_equivalents: float | None = None
    total_liabilities: float | None = None
    current_liabilities: float | None = None
    total_debt: float | None = None
    total_equity: float | None = None

    # Cash Flow Statement
    operating_cash_flow: float | None = None
    capital_expenditure: float | None = None

    # Market Data
    share_price: float | None = None
    shares_outstanding: float | None = None


class FinancialMetrics(BaseModel):
    ticker: str
    fiscal_year: int | str
    currency: str

    # Liquidity Ratios
    current_ratio: float | None = None
    quick_ratio: float | None = None

    # Profitability Ratios
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_profit_margin: float | None = None
    roe: float | None = None
    roa: float | None = None

    # Leverage Ratios
    debt_to_equity: float | None = None
    net_debt: float | None = None
    net_debt_to_ebitda: float | None = None

    # Cash Flow & Valuation
    free_cash_flow: float | None = None
    eps: float | None = None
    pe_ratio: float | None = None

    # Automated Quantitative Red Flags
    red_flags: list[str] = Field(default_factory=list)
