from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from functools import wraps
from itertools import chain
from pathlib import Path
from typing import Awaitable, Callable, Literal, Self

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from evals.apis.inference.openai.utils import get_equivalent_model_ids
from evals.data_models import (
    ChatMessage,
    EmbeddingParams,
    EmbeddingResponseBase64,
    LLMParams,
    LLMResponse,
    MessageRole,
    Prompt,
    TaggedModeration,
)
from evals.utils import get_repo_root, load_secrets, setup_environment

from .anthropic import ANTHROPIC_MODELS, AnthropicChatModel
from .cache_manager import CacheManager
from .huggingface import HUGGINGFACE_MODELS, HuggingFaceModel
from .model import InferenceAPIModel
from .openai.chat import OpenAIChatModel
from .openai.completion import OpenAICompletionModel
from .openai.embedding import OpenAIEmbeddingModel
from .openai.moderation import OpenAIModerationModel
from .openai.utils import COMPLETION_MODELS, GPT_CHAT_MODELS

LOGGER = logging.getLogger(__name__)


_DEFAULT_GLOBAL_INFERENCE_API: InferenceAPI | None = None


class InferenceAPI:
    """
    A wrapper around the OpenAI and Anthropic APIs that automatically manages
    rate limits and valid responses.
    """

    def __init__(
        self,
        anthropic_num_threads: int = 80,
        openai_fraction_rate_limit: float = 0.8,
        openai_num_threads: int = 100,
        organization: str = "ACEDEMICNYUPEREZ_ORG",
        prompt_history_dir: Path | Literal["default"] | None = "default",
        cache_dir: Path | Literal["default"] | None = "default",
    ):
        """
        Set prompt_history_dir to None to disable saving prompt history.
        Set prompt_history_dir to "default" to use the default prompt history directory.
        """

        if openai_fraction_rate_limit >= 1:
            raise ValueError("openai_fraction_rate_limit must be less than 1")

        self.anthropic_num_threads = anthropic_num_threads
        self.openai_fraction_rate_limit = openai_fraction_rate_limit
        self.organization = organization
        # limit openai api calls to 50 to stop jamming
        self.openai_semaphore = asyncio.Semaphore(openai_num_threads)

        if self.organization is None:
            self.organization = "ACEDEMICNYUPEREZ_ORG"

        secrets = load_secrets("SECRETS")
        if prompt_history_dir == "default":
            try:
                self.prompt_history_dir = Path(secrets["PROMPT_HISTORY_DIR"])
            except KeyError:
                self.prompt_history_dir = get_repo_root() / ".prompt_history"
        else:
            assert isinstance(prompt_history_dir, Path) or prompt_history_dir is None
            self.prompt_history_dir = prompt_history_dir

        if cache_dir == "default":
            try:
                self.cache_dir = Path(secrets["CACHE_DIR"])
            except KeyError:
                self.cache_dir = get_repo_root() / ".cache"
        else:
            assert isinstance(cache_dir, Path) or cache_dir is None
            self.cache_dir = cache_dir

        self.cache_manager: CacheManager | None = None
        if self.cache_dir is not None:
            self.cache_manager = CacheManager(self.cache_dir)

        self._openai_completion = OpenAICompletionModel(
            frac_rate_limit=self.openai_fraction_rate_limit,
            organization=secrets[self.organization],
            prompt_history_dir=self.prompt_history_dir,
        )

        if "NYUARG_ORG" in secrets:
            self._openai_gpt4base = OpenAICompletionModel(
                frac_rate_limit=self.openai_fraction_rate_limit,
                organization=secrets["NYUARG_ORG"],
                prompt_history_dir=self.prompt_history_dir,
            )
        else:
            self._openai_gpt4base = self._openai_completion

        self._openai_chat = OpenAIChatModel(
            frac_rate_limit=self.openai_fraction_rate_limit,
            organization=secrets[self.organization],
            prompt_history_dir=self.prompt_history_dir,
        )

        self._openai_moderation = OpenAIModerationModel(organization=secrets[self.organization])

        self._openai_embedding = OpenAIEmbeddingModel(organization=secrets[self.organization])

        self._anthropic_chat = AnthropicChatModel(
            num_threads=self.anthropic_num_threads,
            prompt_history_dir=self.prompt_history_dir,
        )

        self._huggingface = HuggingFaceModel(
            num_threads=100,
            prompt_history_dir=self.prompt_history_dir,
            token=secrets["HF_API_KEY"] if "HF_API_KEY" in secrets else None,
        )

        self.running_cost: float = 0
        self.model_timings = {}
        self.model_wait_times = {}

    def semaphore_method_decorator(func):
        # unused but could be useful to debug if openai jamming comes back
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            async with self.openai_semaphore:
                return await func(self, *args, **kwargs)

        return wrapper

    @classmethod
    def get_default_global_api(cls) -> Self:
        global _DEFAULT_GLOBAL_INFERENCE_API
        if _DEFAULT_GLOBAL_INFERENCE_API is None:
            _DEFAULT_GLOBAL_INFERENCE_API = cls()
        return _DEFAULT_GLOBAL_INFERENCE_API

    @classmethod
    def default_global_running_cost(cls) -> float:
        return cls.get_default_global_api().running_cost

    def model_id_to_class(self, model_id: str) -> InferenceAPIModel:
        if model_id == "gpt-4-base":
            return self._openai_gpt4base  # NYU ARG is only org with access to this model
        elif model_id in COMPLETION_MODELS:
            return self._openai_completion
        elif model_id in GPT_CHAT_MODELS or "ft:gpt-3.5-turbo" in model_id:
            return self._openai_chat
        elif model_id in ANTHROPIC_MODELS:
            return self._anthropic_chat
        elif model_id in HUGGINGFACE_MODELS:
            return self._huggingface
        raise ValueError(f"Invalid model id: {model_id}")

    def filter_responses(
        self,
        candidate_responses: list[LLMResponse],
        n: int,
        is_valid: Callable[[str], bool],
        insufficient_valids_behaviour: Literal["error", "continue", "pad_invalids"] = "error",
    ) -> list[LLMResponse]:
        # filter out invalid responses
        num_candidates = len(candidate_responses)
        valid_responses = [response for response in candidate_responses if is_valid(response.completion)]
        num_valid = len(valid_responses)
        success_rate = num_valid / num_candidates
        if success_rate < 1:
            LOGGER.info(f"`is_valid` success rate: {success_rate * 100:.2f}%")

        # return the valid responses, or pad with invalid responses if there aren't enough
        if num_valid < n:
            match insufficient_valids_behaviour:
                case "error":
                    raise RuntimeError(f"Only found {num_valid} valid responses from {num_candidates} candidates.")
                case "retry":
                    raise RuntimeError(f"Only found {num_valid} valid responses from {num_candidates} candidates.")
                case "continue":
                    responses = valid_responses
                case "pad_invalids":
                    invalid_responses = [
                        response for response in candidate_responses if not is_valid(response.completion)
                    ]
                    invalids_needed = n - num_valid
                    responses = [
                        *valid_responses,
                        *invalid_responses[:invalids_needed],
                    ]
                    LOGGER.info(
                        f"Padded {num_valid} valid responses with {invalids_needed} invalid responses to get {len(responses)} total responses"
                    )
        else:
            responses = valid_responses
        return responses[:n]

    async def __call__(
        self,
        model_ids: str | tuple[str, ...],
        prompt: Prompt,
        max_tokens: int | None = None,
        print_prompt_and_response: bool = False,
        n: int = 1,
        max_attempts_per_api_call: int = 10,
        num_candidates_per_completion: int = 1,
        is_valid: Callable[[str], bool] = lambda _: True,
        insufficient_valids_behaviour: Literal["error", "continue", "pad_invalids", "retry"] = "retry",
        **kwargs,
    ) -> list[LLMResponse]:
        """
        Make maximally efficient API requests for the specified model(s) and prompt.

        Args:
            model_ids: The model(s) to call. If multiple models are specified, the output will be sampled from the
                cheapest model that has capacity. All models must be from the same class (e.g. OpenAI Base,
                OpenAI Chat, or Anthropic Chat). Anthropic chat will error if multiple models are passed in.
                Passing in multiple models could speed up the response time if one of the models is overloaded.
            prompt: The prompt to send to the model(s). Type should be Prompt.
            max_tokens: The maximum number of tokens to request from the API (argument added to
                standardize the Anthropic and OpenAI APIs, which have different names for this).
            print_prompt_and_response: Whether to print the prompt and response to stdout.
            n: The number of completions to request.
            max_attempts_per_api_call: Passed to the underlying API call. If the API call fails (e.g. because the
                API is overloaded), it will be retried this many times. If still fails, an exception will be raised.
            num_candidates_per_completion: How many candidate completions to generate for each desired completion.
                n*num_candidates_per_completion completions will be generated, then is_valid is applied as a filter,
                then the remaining completions are returned up to a maximum of n.
            is_valid: Candiate completions are filtered with this predicate.
            insufficient_valids_behaviour: What should we do if the remaining completions after applying the is_valid
                filter is shorter than n.
                - error: raise an error
                - continue: return the valid responses, even if they are fewer than n
                - pad_invalids: pad the list with invalid responses up to n
                - retry: retry the API call until n valid responses are found
        """

        # trick to double rate limit for most recent model only
        if isinstance(model_ids, str):
            model_ids = get_equivalent_model_ids(model_ids)

        model_classes = [self.model_id_to_class(model_id) for model_id in model_ids]
        if len(set(str(type(x)) for x in model_classes)) != 1:
            raise ValueError("All model ids must be of the same type.")

        # standardize max_tokens argument
        model_class = model_classes[0]
        if isinstance(model_class, AnthropicChatModel):
            max_tokens = max_tokens if max_tokens is not None else 2000
            kwargs["max_tokens"] = max_tokens
        else:
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

        num_candidates = num_candidates_per_completion * n

        llm_cache_params = LLMParams(
            model=model_ids,
            n=n,
            insufficient_valids_behaviour=insufficient_valids_behaviour,
            num_candidates_per_completion=num_candidates_per_completion,
            **kwargs,
        )
        cached_result = None
        if self.cache_manager is not None:
            cached_result = self.cache_manager.maybe_load_cache(prompt=prompt, params=llm_cache_params)
            if cached_result is not None:
                cache_file = self.cache_manager.get_cache_file(prompt=prompt, params=llm_cache_params)
                LOGGER.info(f"Loaded cache for prompt from {cache_file}")

        if cached_result is not None:
            candidate_responses = cached_result.responses
            if insufficient_valids_behaviour != "continue":
                assert len(candidate_responses) == n, f"cache is inconsistent with n={n}\n{candidate_responses}"
            if print_prompt_and_response:
                prompt.pretty_print(candidate_responses)
        elif isinstance(model_class, AnthropicChatModel) or isinstance(model_class, HuggingFaceModel):
            # Anthropic chat doesn't support generating multiple candidates at once, so we have to do it manually
            candidate_responses = list(
                chain.from_iterable(
                    await asyncio.gather(
                        *[
                            model_class(
                                model_ids,
                                prompt,
                                print_prompt_and_response,
                                max_attempts_per_api_call,
                                is_valid=(is_valid if insufficient_valids_behaviour == "retry" else lambda _: True),
                                **kwargs,
                            )
                            for _ in range(num_candidates)
                        ]
                    )
                )
            )
        else:
            async with self.openai_semaphore:
                candidate_responses = await model_class(
                    model_ids,
                    prompt,
                    print_prompt_and_response,
                    max_attempts_per_api_call,
                    n=num_candidates,
                    is_valid=(is_valid if insufficient_valids_behaviour == "retry" else lambda _: True),
                    **kwargs,
                )

        # Save cache
        if self.cache_manager is not None and cached_result is None:
            self.cache_manager.save_cache(
                prompt=prompt,
                params=llm_cache_params,
                responses=candidate_responses,
            )

        # filter based on is_valid criteria and insufficient_valids_behaviour
        responses = self.filter_responses(
            candidate_responses,
            n,
            is_valid,
            insufficient_valids_behaviour,
        )

        if cached_result is None:
            # update running cost and model timings
            self.running_cost += sum(response.cost for response in candidate_responses)
            for response in candidate_responses:
                self.model_timings.setdefault(response.model_id, []).append(response.api_duration)
                self.model_wait_times.setdefault(response.model_id, []).append(
                    response.duration - response.api_duration
                )

        return responses

    async def ask_single_question(
        self,
        model_ids: str | tuple[str, ...],
        question: str,
        system_prompt: str | None = None,
        **api_kwargs,
    ) -> list[str]:
        """Wrapper around __call__ to ask a single question."""
        responses = await self(
            model_ids=model_ids,
            prompt=Prompt(
                messages=(
                    [] if system_prompt is None else [ChatMessage(role=MessageRole.system, content=system_prompt)]
                )
                + [
                    ChatMessage(role=MessageRole.user, content=question),
                ]
            ),
            **api_kwargs,
        )

        return [r.completion for r in responses]

    async def moderate(
        self,
        texts: list[str],
        model_id: str = "text-moderation-latest",
        progress_bar: bool = False,
    ) -> list[TaggedModeration] | None:
        """
        Returns moderation results on text. Free to use with no rate limit, but
        should only be called with outputs from OpenAI models.
        """
        if self.cache_manager is not None:
            cached_result = self.cache_manager.maybe_load_moderation(texts)
            if cached_result is not None:
                return cached_result.moderation

        moderation_result = await self._openai_moderation(
            model_id=model_id,
            texts=texts,
            progress_bar=progress_bar,
        )
        if self.cache_manager is not None:
            self.cache_manager.save_moderation(texts, moderation_result)

        return moderation_result

    async def _embed_single_batch(
        self,
        params: EmbeddingParams,
    ):
        if self.cache_manager is not None:
            cached_result = self.cache_manager.maybe_load_embeddings(params)
            if cached_result is not None:
                return cached_result

        response = await self._openai_embedding.embed(params)

        if self.cache_manager is not None:
            self.cache_manager.save_embeddings(params, response)

        self.running_cost += response.cost
        return response

    async def embed(
        self,
        texts: list[str],
        model_id: str = "text-embedding-3-large",
        dimensions: int | None = None,
        progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Returns embeddings for text.
        TODO: Actually implement a proper rate limit.
        """
        request_futures: list[Awaitable[EmbeddingResponseBase64]] = []

        batch_size = self._openai_embedding.batch_size
        for beg_idx in range(0, len(texts), batch_size):
            batch_texts = texts[beg_idx : beg_idx + batch_size]
            request_futures.append(
                self._embed_single_batch(
                    params=EmbeddingParams(
                        model_id=model_id,
                        texts=batch_texts,
                        dimensions=dimensions,
                    )
                )
            )

        async_lib = tqdm if progress_bar else asyncio
        responses = await async_lib.gather(*request_futures)

        return np.concatenate([r.get_numpy_embeddings() for r in responses])

    def reset_cost(self):
        self.running_cost = 0

    def log_model_timings(self):
        if len(self.model_timings) > 0:
            plt.figure(figsize=(10, 6))
            for model in self.model_timings:
                timings = np.array(self.model_timings[model])
                wait_times = np.array(self.model_wait_times[model])
                LOGGER.info(
                    f"{model}: response {timings.mean():.3f}, waiting {wait_times.mean():.3f} (max {wait_times.max():.3f}, min {wait_times.min():.3f})"
                )
                plt.plot(
                    timings,
                    label=f"{model} - Response Time",
                    linestyle="-",
                    linewidth=2,
                )
                plt.plot(
                    wait_times,
                    label=f"{model} - Waiting Time",
                    linestyle="--",
                    linewidth=2,
                )
            plt.legend()
            plt.title("Model Performance: Response and Waiting Times")
            plt.xlabel("Sample Number")
            plt.ylabel("Time (seconds)")
            plt.savefig(
                self.prompt_history_dir / "model_timings.png",
                bbox_inches="tight",
            )
            plt.close()


async def demo():
    setup_environment()
    model_api = InferenceAPI()

    prompt_examples = [
        [
            {"role": "system", "content": "You are a comedic pirate."},
            {"role": "user", "content": "Hello!"},
        ],
        [
            {
                "role": "system",
                "content": "You are a swashbuckling space-faring voyager.",
            },
            {"role": "user", "content": "Hello!"},
        ],
        [
            {
                "role": "system",
                "content": "You are a swashbuckling space-faring voyager.",
            },
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "How dare you call me a"},
        ],
        [
            {"role": "none", "content": "whence afterever the storm"},
        ],
    ]
    prompts = [Prompt(messages=messages) for messages in prompt_examples]

    # test anthropic chat, and continuing assistant message
    anthropic_requests = [
        model_api(
            "claude-3-opus-20240229",
            prompt=prompts[0],
            n=1,
            print_prompt_and_response=True,
        ),
        model_api(
            "claude-3-opus-20240229",
            prompt=prompts[2],
            n=1,
            print_prompt_and_response=True,
        ),
    ]

    # test OAI chat, more than 1 model and n > 1
    oai_chat_models = ["gpt-3.5-turbo", "gpt-3.5-turbo-0613"]
    oai_chat_requests = [
        model_api(
            oai_chat_models,
            prompt=prompts[1],
            n=6,
            print_prompt_and_response=True,
        ),
    ]

    # test OAI completion with none message and assistant message to continue
    oai_requests = [
        model_api(
            "gpt-3.5-turbo-instruct",
            prompt=prompts[2],
            print_prompt_and_response=True,
        ),
        model_api(
            "gpt-3.5-turbo-instruct",
            prompt=prompts[3],
            print_prompt_and_response=True,
        ),
    ]
    answer = await asyncio.gather(*anthropic_requests, *oai_chat_requests, *oai_requests)

    costs = defaultdict(int)
    for responses in answer:
        for response in responses:
            costs[response.model_id] += response.cost

    print("-" * 80)
    print("Costs:")
    for model_id, cost in costs.items():
        print(f"{model_id}: ${cost}")
    return answer


if __name__ == "__main__":
    asyncio.run(demo())
