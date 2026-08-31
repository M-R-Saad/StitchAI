"""InferenceLog and ReferenceImage table definitions (SQLAlchemy)."""
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class InferenceLog(Base):
    """One row per /infer call, regardless of category — this IS the unified audit
    trail claim (whitepaper Section 2.1 / Phase 5)."""

    __tablename__ = "inference_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String, nullable=False)         # "fabric" | "safety" | "machinery"
    score = Column(Float, nullable=False)
    verdict = Column(String, nullable=False)           # "normal" | "anomalous"
    image_ref = Column(String, nullable=False)         # path or storage key for the image
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ReferenceImage(Base):
    """A single few-shot 'normal' reference image belonging to a category."""

    __tablename__ = "reference_image"

    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String, nullable=False)
    image_path = Column(String, nullable=False)
    added_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
