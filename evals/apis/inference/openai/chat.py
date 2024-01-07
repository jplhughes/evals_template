import logging
import time

import openai
import requests
from openai.openai_object import OpenAIObject as OpenAICompletion
from tenacity import retry, stop_after_attempt, wait_fixed
from termcolor import cprint

from evals.apis.inference.utils import (
    PRINT_COLORS,
    add_response_to_prompt_file,
    create_prompt_history_file,
)
from evals.data_models.language_model import LLMResponse
from evals.apis.inference.openai.base import OpenAIModel
from evals.apis.inference.openai.utils import count_tokens, price_per_token, OAIChatPrompt, GPT_CHAT_MODELS

LOGGER = logging.getLogger(__name__)


class OpenAIChatModel(OpenAIModel):
    def _process_prompt(self, prompt: OAIChatPrompt) -> OAIChatPrompt:
        return prompt

    def _assert_valid_id(self, model_id: str):
        if "ft:" in model_id:
            model_id = model_id.split(":")[1]
        assert model_id in GPT_CHAT_MODELS, f"Invalid model id: {model_id}"

    @retry(stop=stop_after_attempt(8), wait=wait_fixed(2))
    async def _get_dummy_response_header(self, model_id: str):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}",
            "OpenAI-Organization": self.organization,
        }
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Say 1"}],
        }
        response = requests.post(url, headers=headers, json=data)
        if "x-ratelimit-limit-tokens" not in response.headers:
            raise RuntimeError("Failed to get dummy response header")
        return response.headers

    @staticmethod
    def _count_prompt_token_capacity(prompt: OAIChatPrompt, **kwargs) -> int:
        # The magic formula is: .25 * (total number of characters) + (number of messages) + (max_tokens, or 15 if not specified)
        BUFFER = 5  # A bit of buffer for some error margin
        MIN_NUM_TOKENS = 20

        num_tokens = 0
        for message in prompt:
            num_tokens += 1
            num_tokens += len(message["content"]) / 4

        return max(
            MIN_NUM_TOKENS,
            int(num_tokens + BUFFER) + kwargs.get("n", 1) * kwargs.get("max_tokens", 15),
        )

    @staticmethod
    def convert_top_logprobs(data):
        # Initialize the new structure with only top_logprobs
        top_logprobs = []

        for item in data["content"]:
            # Prepare a dictionary for top_logprobs
            top_logprob_dict = {}
            for top_logprob in item["top_logprobs"]:
                top_logprob_dict[top_logprob["token"]] = top_logprob["logprob"]

            top_logprobs.append(top_logprob_dict)

        return top_logprobs

    async def _make_api_call(self, prompt: OAIChatPrompt, model_id, start_time, **params) -> list[LLMResponse]:
        LOGGER.debug(f"Making {model_id} call with {self.organization}")

        if "logprobs" in params:
            params["top_logprobs"] = params["logprobs"]
            params["logprobs"] = True

        prompt_file = create_prompt_history_file(prompt, model_id, self.prompt_history_dir)
        api_start = time.time()
        api_response: OpenAICompletion = await openai.ChatCompletion.acreate(messages=prompt, model=model_id, organization=self.organization, **params)  # type: ignore
        api_duration = time.time() - api_start
        duration = time.time() - start_time
        context_token_cost, completion_token_cost = price_per_token(model_id)
        context_cost = api_response.usage.prompt_tokens * context_token_cost
        responses = [
            LLMResponse(
                model_id=model_id,
                completion=choice.message.content,
                stop_reason=choice.finish_reason,
                api_duration=api_duration,
                duration=duration,
                cost=context_cost + count_tokens(choice.message.content) * completion_token_cost,
                logprobs=self.convert_top_logprobs(choice.logprobs) if choice.logprobs is not None else None,
            )
            for choice in api_response.choices
        ]
        add_response_to_prompt_file(prompt_file, responses)
        return responses

    @staticmethod
    def _print_prompt_and_response(prompts: OAIChatPrompt, responses: list[LLMResponse]):
        for prompt in prompts:
            role, text = prompt["role"], prompt["content"]
            cprint(f"=={role.upper()}:", "white")
            cprint(text, PRINT_COLORS[role])
        for i, response in enumerate(responses):
            if len(responses) > 1:
                cprint(f"==RESPONSE {i + 1} ({response.model_id}):", "white")
            cprint(response.completion, PRINT_COLORS["assistant"], attrs=["bold"])
        print()
