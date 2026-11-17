"""SQLite database engine and request-scoped sessions."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import PROJECT_ROOT, get_settings


class Base(DeclarativeBase):
    pass


def _database_url() -> str:
    value = get_settings().database_url
    if value.startswith("sqlite:///:"):
        return value
    if value.startswith("sqlite:///"):
        raw_path = value.removeprefix("sqlite:///")
        path = PROJECT_ROOT / raw_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path.as_posix()}"
    return value


engine = create_engine(_database_url(), connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_database() -> None:
    from backend.models.detection_record import DetectionRecord  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    database = SessionLocal()
    try:
        yield database
    finally:
        database.close()
