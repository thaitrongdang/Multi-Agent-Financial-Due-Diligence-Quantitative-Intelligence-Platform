import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from src.api.dependencies import get_hybrid_retriever, get_pdf_parser
from src.api.schemas import IngestMarkdownRequest, IngestResponse
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.parser import FinancialPDFParser

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/ingest-text", response_model=IngestResponse)
async def ingest_markdown_document(
    payload: IngestMarkdownRequest,
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    parser: FinancialPDFParser = Depends(get_pdf_parser),
) -> IngestResponse:
    chunks = parser.parse_markdown_sections(
        raw_text=payload.markdown_content,
        ticker=payload.ticker,
        fiscal_year=payload.fiscal_year,
        page_number=payload.page_number,
    )
    indexed = retriever.index_chunks(chunks)
    table_count = sum(1 for c in chunks if c.chunk_type == "table")
    text_count = sum(1 for c in chunks if c.chunk_type == "text")
    return IngestResponse(
        ticker=payload.ticker.strip().upper(),
        chunks_indexed=indexed,
        table_chunks=table_count,
        text_chunks=text_count,
    )


@router.post("/ingest-pdf", response_model=IngestResponse)
async def ingest_pdf_document(
    ticker: str = Form(...),
    fiscal_year: str = Form(...),
    file: UploadFile = File(...),
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    parser: FinancialPDFParser = Depends(get_pdf_parser),
) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    parsed_year: int | str = int(fiscal_year) if fiscal_year.isdigit() else fiscal_year
    content = await file.read()

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)

        chunks = parser.parse_pdf(
            pdf_path=tmp_path,
            ticker=ticker,
            fiscal_year=parsed_year,
        )
        indexed = retriever.index_chunks(chunks)
        table_count = sum(1 for c in chunks if c.chunk_type == "table")
        text_count = sum(1 for c in chunks if c.chunk_type == "text")
        return IngestResponse(
            ticker=ticker.strip().upper(),
            chunks_indexed=indexed,
            table_chunks=table_count,
            text_chunks=text_count,
        )
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
