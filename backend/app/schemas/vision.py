from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=240)]
EvidenceQuality = Literal["strong", "partial", "insufficient"]
EvidenceSource = Literal["original_image", "follow_up_image", "user_reported"]


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: ShortText
    inference: ShortText
    support: Literal["strong", "moderate", "weak"]
    limitation: str = Field(default="", max_length=240)
    source: EvidenceSource = "original_image"


class EvidenceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quality: EvidenceQuality = "insufficient"
    rationale: str = Field(default="", max_length=400)
    supporting_points: list[ShortText] = Field(default_factory=list, max_length=5)
    limitations: list[ShortText] = Field(default_factory=list, max_length=5)


class DistinguishingFeature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: ShortText
    observation: str = Field(max_length=240)
    interpretation: ShortText


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    common_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
    scientific_name: str | None = Field(default=None, max_length=120)
    match_explanation: ShortText
    supporting_features: list[ShortText] = Field(default_factory=list, max_length=5)
    distinguishing_features: list[DistinguishingFeature] = Field(default_factory=list, max_length=5)
    uncertainties: list[ShortText] = Field(default_factory=list, max_length=5)


class ObservationTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: ShortText
    what_to_look_for: ShortText
    why_it_matters: ShortText
    how_safely: ShortText
    possible_outcomes: list[ShortText] = Field(default_factory=list, max_length=3)


class IdentificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["identified", "uncertain", "unknown"] = "uncertain"
    common_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)] = Field(
        description="Likely common name, or 'Unknown organism'"
    )
    scientific_name: str | None = Field(default=None, max_length=120)
    confidence: Literal["low", "moderate", "high"] = "low"
    summary: str = Field(default="", max_length=500)
    visible_evidence: list[ShortText] = Field(default_factory=list, max_length=8)
    evidence_details: list[EvidenceItem] = Field(default_factory=list, max_length=8)
    evidence_assessment: EvidenceAssessment = Field(default_factory=EvidenceAssessment)
    candidates: list[Candidate] = Field(default_factory=list, max_length=3)
    comparison_summary: str | None = Field(default=None, max_length=600)
    interesting_facts: list[ShortText] = Field(default_factory=list, max_length=5)
    observation_tasks: list[ObservationTask] = Field(default_factory=list, max_length=3)
    observe_next: str | None = Field(default=None, max_length=500)
    change_summary: str | None = Field(default=None, max_length=500)
    disclaimer: str = Field(
        default="AI-generated hypothesis only. Verify species and factual claims with a trusted reference.",
        max_length=400,
    )

    @model_validator(mode="after")
    def leading_candidate_matches_identification(self):
        if self.candidates and self.candidates[0].common_name.casefold() != self.common_name.casefold():
            raise ValueError("The first candidate must match the leading identification.")
        return self


class FollowUpResponse(BaseModel):
    identification: IdentificationResult
    what_changed: str | None = Field(default=None, max_length=500)
    remaining_uncertainties: list[ShortText] = Field(max_length=5)


class FollowUpInput(BaseModel):
    previous: IdentificationResult
    observation: str | None = Field(default=None, max_length=800)
