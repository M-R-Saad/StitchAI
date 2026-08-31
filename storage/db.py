"""SQLite/Postgres connection + schema setup."""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from storage.models import Base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./storage/stitchai.db")

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create tables if they don't exist yet. Call once at backend startup."""
    Base.metadata.create_all(bind=engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
