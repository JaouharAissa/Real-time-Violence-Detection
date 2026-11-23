from datetime import datetime
from sqlalchemy import String, DateTime, Float
from sqlalchemy.orm import Mapped, mapped_column

from db import Base


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    source_type: Mapped[str] = mapped_column(String(32), default="upload", index=True)
    source_ref: Mapped[str | None] = mapped_column(String(256), nullable=True)

    label: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    violence_probability: Mapped[float | None] = mapped_column(Float, nullable=True)

