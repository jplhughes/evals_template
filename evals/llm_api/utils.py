import logging
from typing import Protocol
from datetime import datetime
from pathlib import Path
import json

from anthropic import AI_PROMPT, HUMAN_PROMPT
from evals.data_models.language_model import LLMResponse

PRINT_COLORS = {"user": "cyan", "system": "magenta", "assistant": "light_green"}
LOGGER = logging.getLogger(__name__)


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
