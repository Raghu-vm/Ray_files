import json
import logging
from typing import Any, Dict, List

from sqlalchemy import delete, insert, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.models import DocumentMetadata, DocumentPG, DocumentRow
from ...services.embedding_service import embed_document
from ..chunking.semantic_chunker import semantic_chunk
from ..extractors.csv_extractor import extract_csv_rows
from ..extractors.doc_extractor import extract_doc_text
from ..extractors.excel_extractor import extract_excel_rows
from ..extractors.pdf_extractor import extract_pdf_text

logger = logging.getLogger("ray.ingestion")


MIME_PDF = "application/pdf"
MIME_DOC = "application/vnd.google-apps.document"
MIME_SHEET = "application/vnd.google-apps.spreadsheet"
MIME_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
MIME_CSV = "text/csv"


def _schema_from_rows(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "[]"
    return json.dumps(list(rows[0].keys()), ensure_ascii=True)


def _concatenate_rows(rows: List[Dict[str, Any]]) -> str:
    return " ".join(json.dumps(row, ensure_ascii=True, default=str) for row in rows)


class IngestionPipeline:
    def __init__(self, drive_client) -> None:
        self.drive_client = drive_client

    async def ingest_file(self, session: AsyncSession, file_info: Dict[str, Any]) -> None:
        file_id = file_info.get("id") or ""
        file_title = file_info.get("name") or ""
        file_type = file_info.get("mimeType") or ""
        file_url = file_info.get("webViewLink") or ""

        if not file_id:
            logger.warning("Missing file_id; skipping ingestion")
            return

        await self._delete_old_vectors(session, file_id)
        await self._delete_old_rows(session, file_id)
        await self._upsert_document_metadata(session, file_id, file_title, file_url)

        try:
            file_bytes = await self.drive_client.download_file_bytes(file_id, file_type)
        except Exception:
            logger.exception("Failed to download file")
            return

        if file_type == MIME_PDF:
            await self._process_text(session, file_id, file_title, extract_pdf_text(file_bytes))
            return

        if file_type == MIME_DOC:
            await self._process_text(session, file_id, file_title, extract_doc_text(file_bytes))
            return

        if file_type in {MIME_XLSX}:
            rows = extract_excel_rows(file_bytes)
            await self._process_rows(session, file_id, file_title, rows)
            return

        if file_type in {MIME_CSV, MIME_SHEET}:
            rows = extract_csv_rows(file_bytes)
            await self._process_rows(session, file_id, file_title, rows)
            return

        logger.warning("Unsupported mime type: %s", file_type)

    async def _delete_old_vectors(self, session: AsyncSession, file_id: str) -> None:
        await session.execute(
            text(
                """
                DELETE FROM documents_pg
                WHERE metadata->>'file_id' LIKE '%' || :file_id || '%';
                """
            ),
            {"file_id": file_id},
        )
        await session.commit()

    async def _delete_old_rows(self, session: AsyncSession, file_id: str) -> None:
        await session.execute(
            delete(DocumentRow).where(DocumentRow.dataset_id.like(f"%{file_id}%"))
        )
        await session.commit()

    async def _upsert_document_metadata(
        self, session: AsyncSession, file_id: str, title: str, url: str
    ) -> None:
        stmt = pg_insert(DocumentMetadata).values(
            id=file_id,
            title=title,
            url=url,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[DocumentMetadata.id],
            set_={"title": title, "url": url},
        )
        await session.execute(stmt)
        await session.commit()

    async def _update_document_schema(
        self, session: AsyncSession, file_id: str, schema: str
    ) -> None:
        stmt = pg_insert(DocumentMetadata).values(id=file_id, schema=schema)
        stmt = stmt.on_conflict_do_update(
            index_elements=[DocumentMetadata.id],
            set_={"schema": schema},
        )
        await session.execute(stmt)
        await session.commit()

    async def _process_rows(
        self,
        session: AsyncSession,
        file_id: str,
        file_title: str,
        rows: List[Dict[str, Any]],
    ) -> None:
        if rows:
            values = [
                {"dataset_id": file_id, "row_data": row} for row in rows
            ]
            await session.execute(insert(DocumentRow), values)
            await session.commit()

        schema = _schema_from_rows(rows)
        await self._update_document_schema(session, file_id, schema)

        aggregated_text = _concatenate_rows(rows)
        await self._process_text(session, file_id, file_title, aggregated_text)

    async def _process_text(
        self, session: AsyncSession, file_id: str, file_title: str, text_value: str
    ) -> None:
        chunks = await semantic_chunk(text_value)
        if not chunks:
            return

        records = []
        for chunk in chunks:
            content = str(chunk.get("content") or "")
            if not content:
                continue
            try:
                embedding = await embed_document(content)
            except Exception:
                logger.exception("Embedding failed for chunk")
                continue

            metadata = {
                "file_id": file_id,
                "file_title": file_title,
                "chunk": chunk.get("chunk"),
                "chunk_size": chunk.get("chunk_size"),
            }
            records.append(
                {
                    "content": content,
                    "metadata": metadata,
                    "embedding": embedding,
                }
            )

        if records:
            await session.execute(insert(DocumentPG), records)
            await session.commit()
