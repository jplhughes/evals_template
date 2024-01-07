import logging
import time
from typing import Union

import openai
import requests
import tiktoken
from openai.openai_object import OpenAIObject as OpenAICompletion
from tenacity import retry, stop_after_attempt, wait_fixed
from termcolor import cprint

from evals.apis.inference.openai.utils import count_tokens, price_per_token
from evals.apis.inference.utils import (
    PRINT_COLORS,
    add_response_to_prompt_file,
    create_prompt_history_file,
    messages_to_single_prompt,
)
from evals.data_models.language_model import LLMResponse
from evals.apis.inference.openai.base import OpenAIModel
from evals.apis.inference.openai.utils import OAIBasePrompt, OAIChatPrompt, BASE_MODELS

LOGGER = logging.getLogger(__name__)


class OpenAIBaseModel(OpenAIModel):
    def _process_prompt(self, prompt: Union[OAIBasePrompt, OAIChatPrompt]) -> OAIBasePrompt:
        if isinstance(prompt, list) and isinstance(prompt[0], dict):
            return messages_to_single_prompt(prompt)
        return prompt

    def _assert_valid_id(self, model_id: str):
        assert model_id in BASE_MODELS, f"Invalid model id: {model_id}"

    @retry(stop=stop_after_attempt(8), wait=wait_fixed(2))
    async def _get_dummy_response_header(self, model_id: str):
        url = "https://api.openai.com/v1/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}",
            "OpenAI-Organization": self.organization,
        }
        data = {"model": model_id, "prompt": "a", "max_tokens": 1}
        response = requests.post(url, headers=headers, json=data)
        if "x-ratelimit-limit-tokens" not in response.headers:
            raise RuntimeError("Failed to get dummy response header")
        return response.headers

    @staticmethod
    def _count_prompt_token_capacity(prompt: OAIBasePrompt, **kwargs) -> int:
        max_tokens = kwargs.get("max_tokens", 15)
        n = kwargs.get("n", 1)
        completion_tokens = n * max_tokens

        tokenizer = tiktoken.get_encoding("cl100k_base")
        if isinstance(prompt, str):
            prompt_tokens = len(tokenizer.encode(prompt))
            return prompt_tokens + completion_tokens
        else:
            prompt_tokens = sum(len(tokenizer.encode(p)) for p in prompt)
            return prompt_tokens + completion_tokens

    async def _make_api_call(self, prompt: OAIBasePrompt, model_id, start_time, **params) -> list[LLMResponse]:
        LOGGER.debug(f"Making {model_id} call with {self.organization}")
        prompt_file = create_prompt_history_file(prompt, model_id, self.prompt_history_dir)
        api_start = time.time()
        api_response: OpenAICompletion = await openai.Completion.acreate(prompt=prompt, model=model_id, organization=self.organization, **params)  # type: ignore
        api_duration = time.time() - api_start
        duration = time.time() - start_time
        context_token_cost, completion_token_cost = price_per_token(model_id)
        context_cost = api_response.usage.prompt_tokens * context_token_cost
        responses = [
            LLMResponse(
                model_id=model_id,
                completion=choice.text,
                stop_reason=choice.finish_reason,
                api_duration=api_duration,
                duration=duration,
                cost=context_cost + count_tokens(choice.text) * completion_token_cost,
                logprobs=choice.logprobs.top_logprobs if choice.logprobs is not None else None,
            )
            for choice in api_response.choices
        ]
        add_response_to_prompt_file(prompt_file, responses)
        return responses

    @staticmethod
    def _print_prompt_and_response(prompt: OAIBasePrompt, responses: list[LLMResponse]):
        prompt_list = prompt if isinstance(prompt, list) else [prompt]
        responses_per_prompt = len(responses) // len(prompt_list)
        responses_list = [
            responses[i : i + responses_per_prompt] for i in range(0, len(responses), responses_per_prompt)
        ]
        for i, (prompt, response_list) in enumerate(zip(prompt_list, responses_list)):
            if len(prompt_list) > 1:
                cprint(f"==PROMPT {i + 1}", "white")
            if len(response_list) == 1:
                cprint(f"=={response_list[0].model_id}", "white")
                cprint(prompt, PRINT_COLORS["user"], end="")
                cprint(
                    response_list[0].completion,
                    PRINT_COLORS["assistant"],
                    attrs=["bold"],
                )
            else:
                cprint(prompt, PRINT_COLORS["user"])
                for j, response in enumerate(response_list):
                    cprint(f"==RESPONSE {j + 1} ({response.model_id}):", "white")
                    cprint(response.completion, PRINT_COLORS["assistant"], attrs=["bold"])
            print()
