from enum import Enum
from typing import Literal

import openai
import pydantic

from .hashable import HashableBaseModel


class LLMParams(HashableBaseModel):
    model: str | tuple[str, ...]
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    num_candidates_per_completion: int = 1
    insufficient_valids_behaviour: Literal["error", "continue", "pad_invalids", "retry"] = "retry"
    max_tokens: int | None = None
    logprobs: int | None = None

    seed: int | None = None

    model_config = pydantic.ConfigDict(extra="forbid")
    # TODO: Add support for logit_bias


class StopReason(Enum):
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    CONTENT_FILTER = "content_filter"

    def __str__(self):
        return self.value


class LLMResponse(pydantic.BaseModel):
    model_id: str
    completion: str
    stop_reason: StopReason
    cost: float
    duration: float | None = None
    api_duration: float | None = None
    logprobs: list[dict[str, float]] | None = None

    model_config = pydantic.ConfigDict(protected_namespaces=())

    @pydantic.field_validator("stop_reason", mode="before")
    def parse_stop_reason(cls, v: str):
        if v in ["length", "max_tokens"]:
            return StopReason.MAX_TOKENS
        elif v in ["stop", "stop_sequence", "end_turn"]:
            # end_turn is for Anthropic
            return StopReason.STOP_SEQUENCE
        elif v in ["content_filter"]:
            return StopReason.CONTENT_FILTER
        raise ValueError(f"Invalid stop reason: {v}")

    def to_dict(self):
        return {**self.model_dump(), "stop_reason": str(self.stop_reason)}


class TaggedModeration(openai._models.BaseModel):
    # The model used to generate the moderation results.
    model_id: str

    # Actual moderation results
    moderation: openai.types.Moderation

    model_config = pydantic.ConfigDict(protected_namespaces=())
