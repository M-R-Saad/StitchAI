"""Request/response data models for the backend API."""
from typing import List, Optional

from pydantic import BaseModel


class InferenceRequest(BaseModel):
    category: str  # "fabric" | "safety" | "machinery"


class InferenceResponse(BaseModel):
    category: str
    verdict: str            # "normal" | "anomalous"
    score: float
    threshold: float        # the actual threshold applied for this call (from config.yaml)
    heatmap_url: Optional[str] = None
    explanation: Optional[str] = None  # only populated when verdict == "anomalous"
    timestamp: str


class LogEntry(BaseModel):
    id: int
    category: str
    verdict: str
    score: float
    image_ref: str
    timestamp: str


class LogsResponse(BaseModel):
    entries: List[LogEntry]
