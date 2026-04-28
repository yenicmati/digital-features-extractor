from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FeatureStatus(str, Enum):
    LIVE = "Live"
    TO_BE_DEVELOPED = "To Be Developed"
    DEPRECATED = "Deprecated"
    TO_REVIEW = "To Review"


class DigitalFeature(BaseModel):
    id: str
    name: str
    description: str
    status: FeatureStatus = FeatureStatus.TO_REVIEW
    parent_product: str
    entry_points: list[str]
    business_capability_hint: Optional[str] = None
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    source: str
    features: list[DigitalFeature]
    total_clusters: int
    skipped_clusters: int
