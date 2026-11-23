from datetime import datetime
from pydantic import BaseModel


class AlertItem(BaseModel):
    id: str
    createdAt: str
    sourceType: str
    label: str
    confidence: float


class HourlyActivityPoint(BaseModel):
    hour: int
    total: int
    violent: int


class DashboardStats(BaseModel):
    totalAnalyses: int
    violenceRate: float
    alertsToday: int
    activityByHour: list[HourlyActivityPoint]
    recentAlerts: list[AlertItem]


def to_alert_item(a) -> AlertItem:
    return AlertItem(
        id=str(a.id),
        createdAt=a.created_at.isoformat() + "Z",
        sourceType=a.source_type,
        label=a.label,
        confidence=float(a.confidence),
    )

