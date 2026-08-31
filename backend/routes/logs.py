"""GET /logs -> audit/compliance log retrieval. Returns every logged inference,
regardless of category - this IS the unified audit trail claim made real."""
from fastapi import APIRouter

from backend.schemas import LogEntry, LogsResponse
from storage.db import SessionLocal
from storage.models import InferenceLog

router = APIRouter()


@router.get("/logs", response_model=LogsResponse)
async def get_logs(limit: int = 100):
    db = SessionLocal()
    try:
        rows = (
            db.query(InferenceLog)
            .order_by(InferenceLog.timestamp.desc())
            .limit(limit)
            .all()
        )
        entries = [
            LogEntry(
                id=row.id,
                category=row.category,
                verdict=row.verdict,
                score=row.score,
                image_ref=row.image_ref,
                timestamp=row.timestamp.isoformat(),
            )
            for row in rows
        ]
        return LogsResponse(entries=entries)
    finally:
        db.close()
