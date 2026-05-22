import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import anyio
from google.oauth2 import credentials as user_credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build

from ..config import get_drive_watcher_config_path, get_google_drive_credentials
from ..db.session import get_session
from .pipelines.ingestion_pipeline import IngestionPipeline

logger = logging.getLogger("ray.drive")


DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class DriveClient:
    def __init__(self) -> None:
        creds_info = get_google_drive_credentials()
        self.credentials = self._build_credentials(creds_info)
        self.service = build("drive", "v3", credentials=self.credentials, cache_discovery=False)

    def _build_credentials(self, info: Dict[str, Any]):
        if info.get("type") == "service_account":
            return service_account.Credentials.from_service_account_info(
                info, scopes=DRIVE_SCOPES
            )
        return user_credentials.Credentials(
            token=info.get("token"),
            refresh_token=info.get("refresh_token"),
            token_uri=info.get("token_uri"),
            client_id=info.get("client_id"),
            client_secret=info.get("client_secret"),
            scopes=DRIVE_SCOPES,
        )

    def get_start_page_token(self) -> str:
        response = self.service.changes().getStartPageToken().execute()
        return response.get("startPageToken")

    def list_changes(self, page_token: str) -> Dict[str, Any]:
        return (
            self.service.changes()
            .list(
                pageToken=page_token,
                spaces="drive",
                fields="newStartPageToken, changes(fileId, file(id, name, mimeType, modifiedTime, createdTime, webViewLink, parents, trashed))",
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            )
            .execute()
        )

    def get_file_metadata(self, file_id: str) -> Dict[str, Any]:
        return (
            self.service.files()
            .get(
                fileId=file_id,
                fields="id, name, mimeType, webViewLink, parents, modifiedTime, createdTime",
                supportsAllDrives=True,
            )
            .execute()
        )

    def download_file_bytes(self, file_id: str, mime_type: str) -> bytes:
        if mime_type == "application/vnd.google-apps.document":
            request = self.service.files().export(fileId=file_id, mimeType="text/plain")
        elif mime_type == "application/vnd.google-apps.spreadsheet":
            request = self.service.files().export(fileId=file_id, mimeType="text/csv")
        else:
            request = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)

        file_bytes = request.execute()
        return file_bytes


class DriveWatcher:
    def __init__(self) -> None:
        self._config_path = get_drive_watcher_config_path()
        self._config = self._load_config()
        self._client = DriveClient()
        self._pipeline = IngestionPipeline(self._client)
        self._task: asyncio.Task | None = None
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._stopped.set()
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def run_once(self) -> None:
        await self._poll_changes()

    def _load_config(self) -> Dict[str, Any]:
        path = Path(self._config_path)
        if not path.exists():
            raise RuntimeError("drive_watcher_config.json not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_config(self) -> None:
        Path(self._config_path).write_text(
            json.dumps(self._config, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    async def _run(self) -> None:
        poll_seconds = int(self._config.get("poll_interval_seconds") or 60)
        while not self._stopped.is_set():
            try:
                await self._poll_changes()
            except Exception:
                logger.exception("Drive polling failed")
            await asyncio.sleep(poll_seconds)

    async def _poll_changes(self) -> None:
        folder_id = self._config.get("folder_id")
        if not folder_id:
            logger.warning("Missing folder_id in drive_watcher_config.json")
            return

        page_token = self._config.get("state", {}).get("page_token")
        if not page_token:
            page_token = await anyio.to_thread.run_sync(self._client.get_start_page_token)
            self._config.setdefault("state", {})["page_token"] = page_token
            self._save_config()
            return

        response = await anyio.to_thread.run_sync(self._client.list_changes, page_token)
        changes = response.get("changes", []) or []
        new_token = response.get("newStartPageToken")

        file_ids: List[str] = []
        for change in changes:
            file_info = change.get("file") or {}
            if not file_info or file_info.get("trashed"):
                continue
            parents = file_info.get("parents") or []
            if folder_id not in parents:
                continue
            file_ids.append(file_info.get("id"))

        for file_id in file_ids:
            if not file_id:
                continue
            try:
                metadata = await anyio.to_thread.run_sync(
                    self._client.get_file_metadata, file_id
                )
                async with get_session() as session:
                    await self._pipeline.ingest_file(session, metadata)
            except Exception:
                logger.exception("Failed to ingest file: %s", file_id)

        if new_token:
            self._config.setdefault("state", {})["page_token"] = new_token
            self._config["state"]["last_polled_at"] = int(asyncio.get_event_loop().time())
            self._save_config()
