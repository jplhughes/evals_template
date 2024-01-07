import logging
from enum import Enum, auto
from typing import Dict, List, Optional, Protocol
from datetime import datetime
from pathlib import Path
import json

import attrs
from anthropic import AI_PROMPT, HUMAN_PROMPT
from pydantic import BaseModel

PRINT_COLORS = {"user": "cyan", "system": "magenta", "assistant": "light_green"}
LOGGER = logging.getLogger(__name__)


class PromptConfig(BaseModel):
    partials: Dict[str, str] = {}
    word_limit: Optional[int] = 100
    messages: List[Dict[str, str]] = []
    messages_followup: List[Dict[str, str]] = []
    vars: Dict[str, str] = {}


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
    MAX_TOKENS = auto()
    STOP_SEQUENCE = auto()

    @classmethod
    def factory(cls, stop_reason: str) -> "StopReason":
        """
        Parses the openai and anthropic stop reasons into a StopReason enum.
        """
        if stop_reason in ["max_tokens", "length"]:
            return cls.MAX_TOKENS
        elif stop_reason in ["stop_sequence", "stop"]:
            return cls.STOP_SEQUENCE
        raise ValueError(f"Invalid stop reason: {stop_reason}")

    def __repr__(self):
        return self.name


@attrs.frozen()
class LLMResponse:
    model_id: str
    completion: str
    stop_reason: StopReason = attrs.field(converter=StopReason.factory)
    cost: float
    duration: Optional[float] = None
    api_duration: Optional[float] = None
    logprobs: Optional[list[dict[str, float]]] = None

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "completion": self.completion,
            "stop_reason": self.stop_reason.__repr__(),  # Convert to some JSON-serializable format.
            "duration": self.duration,
            "api_duration": self.api_duration,
            "cost": self.cost,
            "logprobs": self.logprobs,
        }


class ModelAPIProtocol(Protocol):
    async def __call__(
        self,
        model_ids: list[str],
        prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        **kwargs,
    ) -> list[LLMResponse]:
        raise NotImplementedError


def messages_to_single_prompt(messages) -> str:
    if len(messages) >= 2 and messages[0]["role"] == "system" and messages[1]["role"] == "user":
        combined_content = messages[0]["content"] + " " + messages[1]["content"]
        messages = [{"role": "user", "content": combined_content}] + messages[2:]
    prompt = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        tag = AI_PROMPT if role == "assistant" else HUMAN_PROMPT
        prompt += f"{tag} {content}"
    if tag != AI_PROMPT:
        prompt += f"{AI_PROMPT}"
    return prompt.strip()


def add_assistant_message(messages: list[dict], assistant_message: str):
    last_role = messages[-1]["role"]
    if last_role == "assistant":
        messages[-1]["content"] += assistant_message
    else:
        messages.append({"role": "assistant", "content": assistant_message})
    return messages


def create_prompt_history_file(prompt: dict, model: str, prompt_history_dir: Path):
    filename = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}.txt"
    prompt_file = prompt_history_dir / model / filename
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    with open(prompt_file, "w") as f:
        json_str = json.dumps(prompt, indent=4)
        json_str = json_str.replace("\\n", "\n")
        f.write(json_str)

    return prompt_file


def add_response_to_prompt_file(prompt_file, responses):
    with open(prompt_file, "a") as f:
        f.write("\n\n======RESPONSE======\n\n")
        json_str = json.dumps([response.to_dict() for response in responses], indent=4)
        json_str = json_str.replace("\\n", "\n")
        f.write(json_str)
