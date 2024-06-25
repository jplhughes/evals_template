import asyncio
import logging
import time
from pathlib import Path
from traceback import format_exc

import anthropic.types
from anthropic import AsyncAnthropic

from evals.data_models import LLMResponse, Prompt

from .model import InferenceAPIModel

ANTHROPIC_MODELS = {
    "claude-instant-1",
    # claude-2 models are deprecated per https://constellationberkeley.slack.com/archives/C06C98ZV8AK/p1710275805552209
    # "claude-2.0",
    # "claude-v1.3",
    # "claude-2.1",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
}
LOGGER = logging.getLogger(__name__)


class AnthropicChatModel(InferenceAPIModel):
    def __init__(
        self,
        num_threads: int,
        prompt_history_dir: Path | None = None,
    ):
        self.num_threads = num_threads
        self.prompt_history_dir = prompt_history_dir
        self.aclient = AsyncAnthropic()  # Assuming AsyncAnthropic has a default constructor
        self.available_requests = asyncio.BoundedSemaphore(int(self.num_threads))

    async def __call__(
        self,
        model_ids: tuple[str, ...],
        prompt: Prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        is_valid=lambda x: True,
        **kwargs,
    ) -> list[LLMResponse]:
        start = time.time()
        assert len(model_ids) == 1, "Anthropic implementation only supports one model at a time."

        (model_id,) = model_ids
        sys_prompt, chat_messages = prompt.anthropic_format()
        prompt_file = self.create_prompt_history_file(prompt, model_id, self.prompt_history_dir)

        LOGGER.debug(f"Making {model_id} call")
        response: anthropic.types.Message | None = None
        duration = None
        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    api_start = time.time()

                    response = await self.aclient.messages.create(
                        messages=chat_messages,
                        model=model_id,
                        **kwargs,
                        **(dict(system=sys_prompt) if sys_prompt else {}),
                    )

                    api_duration = time.time() - api_start
                    if not is_valid(response):
                        raise RuntimeError(f"Invalid response according to is_valid {response}")
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

        assert len(response.content) == 1  # Anthropic doesn't support multiple completions (as of 2024-03-12).

        response = LLMResponse(
            model_id=model_id,
            completion=response.content[0].text,
            stop_reason=response.stop_reason,
            duration=duration,
            api_duration=api_duration,
            cost=0,
        )
        responses = [response]

        self.add_response_to_prompt_file(prompt_file, responses)
        if print_prompt_and_response:
            prompt.pretty_print(responses)

        return responses

    def make_stream_api_call(
        self,
        model_id: str,
        prompt: Prompt,
        max_tokens: int,
        **params,
    ) -> anthropic.AsyncMessageStreamManager[anthropic.AsyncMessageStream]:
        # TODO(tony): Can eventually wrap this in an async generator and keep
        #             track of cost. Will need to do this in the parent API
        #             class.
        sys_prompt, chat_messages = prompt.anthropic_format()
        return self.aclient.messages.stream(
            model=model_id,
            messages=chat_messages,
            **(dict(system=sys_prompt) if sys_prompt else {}),
            max_tokens=max_tokens,
            **params,
        )
