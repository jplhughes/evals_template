import asyncio
import logging
import re
import time
from pathlib import Path
from traceback import format_exc
from typing import Optional, Union

import attrs
from anthropic import AsyncAnthropic
from anthropic.types.completion import Completion as AnthropicCompletion
from termcolor import cprint

from evals.data_models.language_model import LLMResponse
from evals.llm_api.openai import OAIChatPrompt
from evals.llm_api.utils import (
    PRINT_COLORS,
    ModelAPIProtocol,
    add_response_to_prompt_file,
    create_prompt_history_file,
    messages_to_single_prompt,
)

ANTHROPIC_MODELS = {"claude-instant-1", "claude-2.0", "claude-v1.3", "claude-2.1"}
LOGGER = logging.getLogger(__name__)


def count_tokens(prompt: str) -> int:
    return len(prompt.split())


def price_per_token(model_id: str) -> tuple[float, float]:
    """
    Returns the (input token, output token) price for the given model id.
    """
    return 0, 0


@attrs.define()
class AnthropicChatModel(ModelAPIProtocol):
    num_threads: int
    print_prompt_and_response: bool = False
    prompt_history_dir: Path = Path("./prompt_history")
    client: AsyncAnthropic = attrs.field(init=False, default=attrs.Factory(AsyncAnthropic))
    available_requests: asyncio.BoundedSemaphore = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.available_requests = asyncio.BoundedSemaphore(int(self.num_threads))

    async def __call__(
        self,
        model_ids: list[str],
        prompt: Union[str, OAIChatPrompt],
        print_prompt_and_response: bool,
        max_attempts: int,
        **kwargs,
    ) -> list[LLMResponse]:
        start = time.time()
        assert len(model_ids) == 1, "Anthropic implementation only supports one model at a time."
        model_id = model_ids[0]
        if isinstance(prompt, list):
            prompt = messages_to_single_prompt(prompt)

        prompt_file = create_prompt_history_file(prompt, model_id, self.prompt_history_dir)
        LOGGER.debug(f"Making {model_id} call")
        response: Optional[AnthropicCompletion] = None
        duration = None
        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    api_start = time.time()
                    response = await self.client.completions.create(prompt=prompt, model=model_id, **kwargs)
                    api_duration = time.time() - api_start
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                await asyncio.sleep(1.5**i)
            else:
                break

        if response is None:
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")

        num_context_tokens, num_completion_tokens = await asyncio.gather(
            self.client.count_tokens(prompt),
            self.client.count_tokens(response.completion),
        )
        context_token_cost, completion_token_cost = price_per_token(model_id)
        cost = num_context_tokens * context_token_cost + num_completion_tokens * completion_token_cost
        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        llm_response = LLMResponse(
            model_id=model_id,
            completion=response.completion,
            stop_reason=response.stop_reason,
            duration=duration,
            api_duration=api_duration,
            cost=cost,
        )

        add_response_to_prompt_file(prompt_file, [llm_response])
        if self.print_prompt_and_response or print_prompt_and_response:
            cprint(prompt, "yellow")
            pattern = r"(Human: |Assistant: )(.*?)(?=(Human: |Assistant: )|$)"
            for match in re.finditer(pattern, prompt, re.S):  # re.S makes . match any character, including a newline
                role = match.group(1).removesuffix(": ").lower()
                role = {"human": "user"}.get(role, role)
                cprint(match.group(2), PRINT_COLORS[role])
            cprint(f"Response ({llm_response.model_id}):", "white")
            cprint(f"{llm_response.completion}", PRINT_COLORS["assistant"], attrs=["bold"])
            print()

        return [llm_response]
