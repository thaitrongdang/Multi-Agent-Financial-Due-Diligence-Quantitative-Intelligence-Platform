from typing import Any

import httpx
import plotly.graph_objects as go
import streamlit as st

from src.ui.client import (
    DEFAULT_API_URL,
    DEMO_PRESETS,
    check_backend_health,
    ingest_pdf_document,
    ingest_text_document,
    stream_due_diligence,
)

st.set_page_config(
    page_title="FinIntel AI | Multi-Agent Due Diligence",
    page_icon="📊",
    layout="wide",
)


def _render_ratios_chart(metrics: dict[str, Any]) -> go.Figure:
    labels = ["Current Ratio", "Quick Ratio", "Debt / Equity", "Net Debt / EBITDA"]
    keys = ["current_ratio", "quick_ratio", "debt_to_equity", "net_debt_to_ebitda"]
    values = [float(metrics.get(k) or 0.0) for k in keys]
    colors = ["#2563eb", "#0ea5e9", "#f59e0b", "#ef4444"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=[f"{v:.2f}x" for v in values],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Liquidity & Leverage Multiples",
        yaxis_title="Multiple (x)",
        height=320,
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
    )
    return fig


def _render_margins_chart(metrics: dict[str, Any]) -> go.Figure:
    labels = ["Gross Margin", "Operating Margin", "Net Margin", "ROE", "ROA"]
    keys = ["gross_margin", "operating_margin", "net_profit_margin", "roe", "roa"]
    values = [round(float(metrics.get(k) or 0.0) * 100, 2) for k in keys]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color="#10b981",
                text=[f"{v:.1f}%" for v in values],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Profitability & Return Profile (%)",
        yaxis_title="Percentage (%)",
        height=320,
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
    )
    return fig


def main() -> None:
    st.title("📊 Multi-Agent Financial Due Diligence & Quantitative Intelligence")
    st.caption(
        "LangGraph Cyclic StateGraph • Deterministic Quant Engine • "
        "Table-Aware Hybrid RAG • Critic Fact-Checking Guardrail"
    )

    with st.sidebar:
        st.header("⚙️ Execution Configuration")
        api_url = st.text_input("FastAPI Backend URL", value=DEFAULT_API_URL)
        is_online = check_backend_health(api_url)
        if is_online:
            st.success("Backend API: Online")
        else:
            st.error("Backend API: Offline (Run uvicorn on port 8000)")

        mode = st.radio(
            "Data Source Mode",
            options=["Interactive Benchmark Preset", "Live Market Data (yfinance)"],
        )

        preset_data: dict[str, Any] | None = None
        if mode == "Interactive Benchmark Preset":
            preset_name = st.selectbox("Select Case Study", list(DEMO_PRESETS.keys()))
            preset_data = DEMO_PRESETS[preset_name]
            ticker = st.text_input("Ticker", value=preset_data["ticker"])
            fiscal_year = st.number_input(
                "Fiscal Year", value=int(preset_data["fiscal_year"]), step=1
            )
            query = st.text_area("Due Diligence Scope", value=preset_data["query"], height=90)
        else:
            ticker = st.text_input("Ticker Symbol", value="MSFT")
            fiscal_year = st.number_input("Fiscal Year", value=2025, step=1)
            query = st.text_area(
                "Due Diligence Scope",
                value="Comprehensive financial health, leverage, and liquidity assessment.",
                height=90,
            )

        uploaded_pdf = st.file_uploader("Optional Filing PDF", type=["pdf"])
        run_btn = st.button("🚀 Run Multi-Agent Due Diligence", type="primary", width="stretch")

    if not run_btn:
        st.info(
            "Configure a ticker or choose a preset in the sidebar, then click "
            "**Run Multi-Agent Due Diligence** to watch the agents execute in real time."
        )
        return

    if not is_online:
        st.error(
            "Cannot connect to FastAPI server. Start it with: "
            "`uv run uvicorn src.api.main:app --reload`"
        )
        return

    try:
        if uploaded_pdf is not None:
            with st.spinner("Parsing tables and indexing PDF into Hybrid Retriever..."):
                ingest_res = ingest_pdf_document(
                    api_base_url=api_url,
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                    filename=uploaded_pdf.name,
                    file_bytes=uploaded_pdf.getvalue(),
                )
                st.toast(
                    f"Indexed {ingest_res['chunks_indexed']} chunks "
                    f"({ingest_res['table_chunks']} tables)"
                )
        elif preset_data and preset_data.get("filing_markdown"):
            ingest_text_document(
                api_base_url=api_url,
                ticker=ticker,
                fiscal_year=fiscal_year,
                markdown_content=preset_data["filing_markdown"],
                page_number=14,
            )

        request_payload: dict[str, Any] = {
            "ticker": ticker,
            "fiscal_year": int(fiscal_year),
            "query": query,
            "max_retries": 2,
        }
        if preset_data is not None:
            request_payload["financial_input"] = preset_data["financial_input"]
            request_payload["news_headlines"] = preset_data["news_headlines"]

        completed_data: dict[str, Any] | None = None
        with st.status("Executing LangGraph Multi-Agent Workflow...", expanded=True) as status:
            for event in stream_due_diligence(api_url, request_payload):
                ev_type = event.get("event")
                if ev_type == "node_update":
                    node = event.get("node", "unknown").upper()
                    traces = event.get("trace", [])
                    detail = traces[-1] if traces else f"Completed {node}"
                    st.write(f"✅ **{node}** — {detail}")
                elif ev_type == "error":
                    status.update(label="Analysis Failed", state="error")
                    st.error(event.get("detail", "Unknown backend error"))
                    return
                elif ev_type == "completed":
                    completed_data = event
                    status.update(label="Due Diligence Completed & Audited", state="complete")

        if completed_data is None:
            st.warning("Stream ended without a final report payload.")
            return

        metrics = completed_data.get("financial_metrics", {})
        is_faithful = completed_data.get("is_faithful", False)
        retries = completed_data.get("retry_count", 0)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Ratio", f"{metrics.get('current_ratio') or 'N/A'}")
        col2.metric("Debt / Equity", f"{metrics.get('debt_to_equity') or 'N/A'}")
        roe_val = metrics.get("roe")
        col3.metric("ROE", f"{roe_val * 100:.2f}%" if roe_val is not None else "N/A")
        guardrail_label = "VERIFIED" if is_faithful else "FLAGGED"
        col4.metric("Critic Guardrail", guardrail_label, delta=f"Retries: {retries}")

        red_flags = metrics.get("red_flags", [])
        if red_flags:
            st.error(
                "🚨 **Automated Quantitative Red Flags Detected:**\n\n"
                + "\n".join(f"- {flag}" for flag in red_flags)
            )

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.plotly_chart(_render_ratios_chart(metrics), width="stretch")
        with chart_col2:
            st.plotly_chart(_render_margins_chart(metrics), width="stretch")

        st.subheader("📑 Institutional Investment Due Diligence Memo")
        st.markdown(completed_data.get("draft_report", ""))

        with st.expander("🔍 View Retrieved Filing Evidence & Reranker Scores"):
            contexts = completed_data.get("extracted_contexts", [])
            if not contexts:
                st.write("No external filing chunks were indexed for this ticker.")
            for ctx in contexts:
                st.markdown(
                    f"**Chunk `{ctx['chunk_id']}`** (Page {ctx['page_number']} • "
                    f"Type: `{ctx['chunk_type']}` • Rerank Score: `{ctx['rerank_score']}`)"
                )
                st.code(ctx["content"], language="markdown")

    except httpx.HTTPError as exc:
        st.error(f"API Communication Error: {exc}")


if __name__ == "__main__":
    main()
