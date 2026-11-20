from src.quant.schemas import FinancialMetrics, FinancialStatementInput


def _safe_div(
    numerator: float | None,
    denominator: float | None,
    decimals: int = 4,
) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return round(numerator / denominator, decimals)


def _detect_red_flags(
    data: FinancialStatementInput,
    current_ratio: float | None,
    debt_to_equity: float | None,
    net_debt_to_ebitda: float | None,
) -> list[str]:
    flags: list[str] = []

    if current_ratio is not None and current_ratio < 1.0:
        flags.append(
            f"Liquidity risk: Current Ratio is {current_ratio:.2f} (< 1.0), "
            "indicating short-term obligations exceed current assets."
        )

    if debt_to_equity is not None and debt_to_equity > 2.0:
        flags.append(
            f"High leverage: Debt-to-Equity is {debt_to_equity:.2f} (> 2.0), "
            "signaling heavy reliance on debt financing."
        )

    if net_debt_to_ebitda is not None and net_debt_to_ebitda > 3.5:
        flags.append(f"Debt service strain: Net Debt/EBITDA is {net_debt_to_ebitda:.2f}x (> 3.5x).")

    if (
        data.net_income is not None
        and data.operating_cash_flow is not None
        and data.net_income > 0
        and data.operating_cash_flow < 0
    ):
        flags.append(
            "Earnings quality warning: Positive Net Income accompanied by "
            "negative Operating Cash Flow."
        )

    return flags


def calculate_financial_metrics(data: FinancialStatementInput) -> FinancialMetrics:
    gross_profit = data.gross_profit
    if gross_profit is None and data.revenue is not None and data.cost_of_revenue is not None:
        gross_profit = data.revenue - data.cost_of_revenue

    current_ratio = _safe_div(data.current_assets, data.current_liabilities)

    quick_assets: float | None = None
    if data.current_assets is not None:
        quick_assets = data.current_assets - (data.inventory or 0.0)
    quick_ratio = _safe_div(quick_assets, data.current_liabilities)

    gross_margin = _safe_div(gross_profit, data.revenue)
    operating_margin = _safe_div(data.operating_income, data.revenue)
    net_profit_margin = _safe_div(data.net_income, data.revenue)
    roe = _safe_div(data.net_income, data.total_equity)
    roa = _safe_div(data.net_income, data.total_assets)

    debt_to_equity = _safe_div(data.total_debt, data.total_equity)

    net_debt: float | None = None
    if data.total_debt is not None:
        net_debt = round(data.total_debt - (data.cash_and_equivalents or 0.0), 2)

    net_debt_to_ebitda: float | None = None
    if net_debt is not None and data.ebitda is not None and data.ebitda > 0:
        net_debt_to_ebitda = _safe_div(net_debt, data.ebitda)

    free_cash_flow: float | None = None
    if data.operating_cash_flow is not None and data.capital_expenditure is not None:
        free_cash_flow = round(data.operating_cash_flow - abs(data.capital_expenditure), 2)

    eps = _safe_div(data.net_income, data.shares_outstanding)
    pe_ratio: float | None = None
    if eps is not None and eps > 0:
        pe_ratio = _safe_div(data.share_price, eps)

    red_flags = _detect_red_flags(
        data=data,
        current_ratio=current_ratio,
        debt_to_equity=debt_to_equity,
        net_debt_to_ebitda=net_debt_to_ebitda,
    )

    return FinancialMetrics(
        ticker=data.ticker.upper(),
        fiscal_year=data.fiscal_year,
        currency=data.currency,
        current_ratio=current_ratio,
        quick_ratio=quick_ratio,
        gross_margin=gross_margin,
        operating_margin=operating_margin,
        net_profit_margin=net_profit_margin,
        roe=roe,
        roa=roa,
        debt_to_equity=debt_to_equity,
        net_debt=net_debt,
        net_debt_to_ebitda=net_debt_to_ebitda,
        free_cash_flow=free_cash_flow,
        eps=eps,
        pe_ratio=pe_ratio,
        red_flags=red_flags,
    )
