from enum import Enum
from pydantic import BaseModel, validator
from typing import Dict, List, Optional


class PromptConfig(BaseModel):
    method: str
    word_limit: Optional[int] = 100
    messages: List[Dict[str, str]] = []
    messages_followup: List[Dict[str, str]] = []
    extra: Dict[str, str] = {}


class LanguageModelConfig(BaseModel):
    model: str
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    max_words: int = 10000
    min_words: int = 0
    num_candidates_per_completion: int = 1
    timeout: int = 120


class StopReason(Enum):
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"

    def __str__(self):
        return self.value


class LLMResponse(BaseModel):
    model_id: str
    completion: str
    stop_reason: StopReason
    cost: float
    duration: Optional[float] = None
    api_duration: Optional[float] = None
    logprobs: Optional[List[dict[str, float]]] = None

    @validator("stop_reason", pre=True)
    def parse_stop_reason(cls, v):
        if v in ["length", "max_tokens"]:
            return StopReason.MAX_TOKENS
        elif v in ["stop", "stop_sequence"]:
            return StopReason.STOP_SEQUENCE
        raise ValueError(f"Invalid stop reason: {v}")

    def to_dict(self):
        return {**self.dict(), "stop_reason": str(self.stop_reason)}
