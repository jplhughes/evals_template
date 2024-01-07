import asyncio
import logging
import time
from pathlib import Path
from traceback import format_exc
from typing import Optional

from anthropic import AsyncAnthropic
from anthropic.types.completion import Completion as AnthropicCompletion

from evals.data_models.messages import Prompt
from evals.apis.inference.utils import (
    InferenceAPIModel,
    add_response_to_prompt_file,
    create_prompt_history_file,
)
from evals.data_models.language_model import LLMResponse

ANTHROPIC_MODELS = {"claude-instant-1", "claude-2.0", "claude-v1.3", "claude-2.1"}
LOGGER = logging.getLogger(__name__)


class AnthropicChatModel(InferenceAPIModel):
    def __init__(self, num_threads, prompt_history_dir=Path("./prompt_history")):
        self.num_threads = num_threads
        self.prompt_history_dir = prompt_history_dir
        self.client = AsyncAnthropic()  # Assuming AsyncAnthropic has a default constructor
        self.available_requests = asyncio.BoundedSemaphore(int(self.num_threads))

    async def __call__(
        self,
        model_ids: list[str],
        prompt: Prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        **kwargs,
    ) -> list[LLMResponse]:
        start = time.time()
        assert len(model_ids) == 1, "Anthropic implementation only supports one model at a time."
        model_id = model_ids[0]
        prompt_string = prompt.anthropic_format()
        prompt_file = create_prompt_history_file(prompt_string, model_id, self.prompt_history_dir)
        LOGGER.debug(f"Making {model_id} call")
        response: Optional[AnthropicCompletion] = None
        duration = None
        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    api_start = time.time()
                    response = await self.client.completions.create(prompt=prompt_string, model=model_id, **kwargs)
                    api_duration = time.time() - api_start
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                await asyncio.sleep(1.5**i)
            else:
                break

        if response is None:
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")

        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        response = LLMResponse(
            model_id=model_id,
            completion=response.completion,
            stop_reason=response.stop_reason,
            duration=duration,
            api_duration=api_duration,
            cost=0,
        )
        responses = [response]

        add_response_to_prompt_file(prompt_file, responses)
        if print_prompt_and_response:
            prompt.pretty_print(responses)

        return responses
