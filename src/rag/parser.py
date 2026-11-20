import re
from pathlib import Path

import pdfplumber

from src.rag.schemas import DocumentChunk


def table_rows_to_markdown(rows: list[list[str | None]]) -> str:
    cleaned_rows: list[list[str]] = []
    for row in rows:
        cleaned = [re.sub(r"\s+", " ", (cell or "").strip()) for cell in row]
        if any(cleaned):
            cleaned_rows.append(cleaned)

    if len(cleaned_rows) < 2:
        return ""

    col_count = max(len(r) for r in cleaned_rows)
    padded = [r + [""] * (col_count - len(r)) for r in cleaned_rows]

    header = "| " + " | ".join(padded[0]) + " |"
    separator = "| " + " | ".join(["---"] * col_count) + " |"
    body = ["| " + " | ".join(r) + " |" for r in padded[1:]]
    return "\n".join([header, separator, *body])


class FinancialPDFParser:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, text: str) -> list[str]:
        normalized = re.sub(r"\n{3,}", "\n\n", text.strip())
        if not normalized:
            return []
        if len(normalized) <= self.chunk_size:
            return [normalized]

        chunks: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(start + self.chunk_size, len(normalized))
            if end < len(normalized):
                split_pos = normalized.rfind(" ", start, end)
                if split_pos > start + (self.chunk_size // 2):
                    end = split_pos
            piece = normalized[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(normalized):
                break
            start = max(end - self.chunk_overlap, start + 1)
        return chunks

    def parse_markdown_sections(
        self,
        raw_text: str,
        ticker: str,
        fiscal_year: int | str,
        page_number: int = 1,
    ) -> list[DocumentChunk]:
        clean_ticker = ticker.strip().upper()
        blocks = re.split(r"\n\s*\n", raw_text.strip())
        results: list[DocumentChunk] = []
        idx = 0

        for block in blocks:
            stripped = block.strip()
            if not stripped:
                continue
            lines = [line.strip() for line in stripped.splitlines() if line.strip()]
            is_table = len(lines) >= 2 and all(
                line.startswith("|") and line.endswith("|") for line in lines
            )

            if is_table:
                results.append(
                    DocumentChunk(
                        chunk_id=f"{clean_ticker}_{fiscal_year}_p{page_number}_t{idx}",
                        ticker=clean_ticker,
                        fiscal_year=fiscal_year,
                        page_number=page_number,
                        chunk_type="table",
                        content=stripped,
                    )
                )
                idx += 1
            else:
                for text_piece in self._split_text(stripped):
                    results.append(
                        DocumentChunk(
                            chunk_id=f"{clean_ticker}_{fiscal_year}_p{page_number}_c{idx}",
                            ticker=clean_ticker,
                            fiscal_year=fiscal_year,
                            page_number=page_number,
                            chunk_type="text",
                            content=text_piece,
                        )
                    )
                    idx += 1
        return results

    def parse_pdf(
        self,
        pdf_path: str | Path,
        ticker: str,
        fiscal_year: int | str,
    ) -> list[DocumentChunk]:
        clean_ticker = ticker.strip().upper()
        chunks: list[DocumentChunk] = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables() or []
                for t_idx, raw_table in enumerate(tables):
                    md_table = table_rows_to_markdown(raw_table)
                    if md_table:
                        chunks.append(
                            DocumentChunk(
                                chunk_id=f"{clean_ticker}_{fiscal_year}_p{page_idx}_t{t_idx}",
                                ticker=clean_ticker,
                                fiscal_year=fiscal_year,
                                page_number=page_idx,
                                chunk_type="table",
                                content=md_table,
                            )
                        )

                raw_text = page.extract_text() or ""
                for c_idx, text_piece in enumerate(self._split_text(raw_text)):
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{clean_ticker}_{fiscal_year}_p{page_idx}_c{c_idx}",
                            ticker=clean_ticker,
                            fiscal_year=fiscal_year,
                            page_number=page_idx,
                            chunk_type="text",
                            content=text_piece,
                        )
                    )
        return chunks
