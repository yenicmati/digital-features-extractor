from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class FeatureStatus(str, Enum):
    LIVE = "Live"
    TO_BE_DEVELOPED = "To Be Developed"
    DEPRECATED = "Deprecated"
    TO_REVIEW = "To Review"


class DigitalFeature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str
    status: FeatureStatus = FeatureStatus.TO_REVIEW
    parent_product: str
    entry_points: list[str]
    business_capability_hint: str | None = None
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    features: list[DigitalFeature]
    total_clusters: int = Field(ge=0)
    skipped_clusters: int = Field(ge=0)


class BusinessFeature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str
    digital_features: list[DigitalFeature]


class GroupingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    business_features: list[BusinessFeature]
    ungrouped_feature_ids: list[str]
