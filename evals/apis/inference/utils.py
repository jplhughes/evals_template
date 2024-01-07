import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Protocol


from evals.data_models.language_model import LLMResponse

LOGGER = logging.getLogger(__name__)


class InferenceAPIModel(Protocol):
    async def __call__(
        self,
        model_ids: list[str],
        prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        **kwargs,
    ) -> list[LLMResponse]:
        raise NotImplementedError


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
