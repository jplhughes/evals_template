import asyncio
import logging
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Callable, Literal, Union

import matplotlib.pyplot as plt
import numpy as np

from evals.apis.inference.anthropic import ANTHROPIC_MODELS, AnthropicChatModel
from evals.apis.inference.openai.chat import OpenAIChatModel
from evals.apis.inference.openai.completion import OpenAICompletionModel
from evals.apis.inference.openai.utils import COMPLETION_MODELS, GPT_CHAT_MODELS
from evals.apis.inference.utils import InferenceAPIModel, LLMResponse
from evals.data_models.messages import Prompt
from evals.utils import load_secrets

LOGGER = logging.getLogger(__name__)


class InferenceAPI:
    def __init__(
        self,
        anthropic_num_threads=5,
        openai_fraction_rate_limit=0.99,
        organization="ACEDEMICNYUPEREZ_ORG",
        exp_dir=Path("./exp"),
    ):
        if openai_fraction_rate_limit >= 1:
            raise ValueError("openai_fraction_rate_limit must be less than 1")

        self.anthropic_num_threads = anthropic_num_threads
        self.openai_fraction_rate_limit = openai_fraction_rate_limit
        self.organization = organization
        self.exp_dir = exp_dir
        self.prompt_history_dir = self.exp_dir / "prompt_history"
        self.prompt_history_dir.mkdir(parents=True, exist_ok=True)

        secrets = load_secrets("SECRETS")
        if self.organization is None:
            self.organization = "ACEDEMICNYUPEREZ_ORG"

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

        self._anthropic_chat = AnthropicChatModel(
            num_threads=self.anthropic_num_threads,
            prompt_history_dir=self.prompt_history_dir,
        )

        self.running_cost = 0
        self.model_timings = {}
        self.model_wait_times = {}

    async def __call__(
        self,
        model_ids: Union[str, list[str]],
        prompt: Prompt,
        max_tokens: int,
        print_prompt_and_response: bool = False,
        n: int = 1,
        max_attempts_per_api_call: int = 10,
        num_candidates_per_completion: int = 1,
        is_valid: Callable[[str], bool] = lambda _: True,
        insufficient_valids_behaviour: Literal["error", "continue", "pad_invalids"] = "error",
        **kwargs,
    ) -> list[LLMResponse]:
        """
        Make maximally efficient API requests for the specified model(s) and prompt.

        Args:
            model_ids: The model(s) to call. If multiple models are specified, the output will be sampled from the
                cheapest model that has capacity. All models must be from the same class (e.g. OpenAI Base,
                OpenAI Chat, or Anthropic Chat). Anthropic chat will error if multiple models are passed in.
                Passing in multiple models could speed up the response time if one of the models is overloaded.
            prompt: The prompt to send to the model(s). Type should match what's expected by the model(s).
            max_tokens: The maximum number of tokens to request from the API (argument added to
                standardize the Anthropic and OpenAI APIs, which have different names for this).
            print_prompt_and_response: Whether to print the prompt and response to stdout.
            n: The number of completions to request.
            max_attempts_per_api_call: Passed to the underlying API call. If the API call fails (e.g. because the
                API is overloaded), it will be retried this many times. If still fails, an exception will be raised.
            num_candidates_per_completion: How many candidate completions to generate for each desired completion. n*num_candidates_per_completion completions will be generated, then is_valid is applied as a filter, then the remaining completions are returned up to a maximum of n.
            is_valid: Candiate completions are filtered with this predicate.
            insufficient_valids_behaviour: What should we do if the remaining completions after applying the is_valid filter is shorter than n.
                error: raise an error
                continue: return the valid responses, even if they are fewer than n
                pad_invalids: pad the list with invalid responses up to n
        """

        assert "max_tokens_to_sample" not in kwargs, "max_tokens_to_sample should be passed in as max_tokens."

        if isinstance(model_ids, str):
            # trick to double rate limit for most recent model only
            if model_ids.endswith("-0613"):
                model_ids = [model_ids, model_ids.replace("-0613", "")]
                print(f"doubling rate limit for most recent model {model_ids}")
            elif model_ids.endswith("-0914"):
                model_ids = [model_ids, model_ids.replace("-0914", "")]
            else:
                model_ids = [model_ids]

        def model_id_to_class(model_id: str) -> InferenceAPIModel:
            if model_id == "gpt-4-base":
                return self._openai_gpt4base  # NYU ARG is only org with access to this model
            elif model_id in COMPLETION_MODELS:
                return self._openai_completion
            elif model_id in GPT_CHAT_MODELS or "ft:gpt-3.5-turbo" in model_id:
                return self._openai_chat
            elif model_id in ANTHROPIC_MODELS:
                return self._anthropic_chat
            raise ValueError(f"Invalid model id: {model_id}")

        model_classes = [model_id_to_class(model_id) for model_id in model_ids]
        if len(set(str(type(x)) for x in model_classes)) != 1:
            raise ValueError("All model ids must be of the same type.")

        # standardize max_tokens argument
        model_class = model_classes[0]
        if isinstance(model_class, AnthropicChatModel):
            max_tokens = max_tokens if max_tokens is not None else 2000
            kwargs["max_tokens_to_sample"] = max_tokens
        else:
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

        num_candidates = num_candidates_per_completion * n
        if isinstance(model_class, AnthropicChatModel):
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
                                **kwargs,
                            )
                            for _ in range(num_candidates)
                        ]
                    )
                )
            )
        else:
            candidate_responses = await model_class(
                model_ids,
                prompt,
                print_prompt_and_response,
                max_attempts_per_api_call,
                n=num_candidates,
                **kwargs,
            )

        # filter out invalid responses
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
                case "continue":
                    responses = valid_responses
                case "pad_invalids":
                    invalid_responses = [
                        response for response in candidate_responses if not is_valid(response.completion)
                    ]
                    invalids_needed = n - num_valid
                    responses = [*valid_responses, *invalid_responses[:invalids_needed]]
                    LOGGER.info(
                        f"Padded {num_valid} valid responses with {invalids_needed} invalid responses to get {len(responses)} total responses"
                    )
        else:
            responses = valid_responses

        # update running cost and model timings
        self.running_cost += sum(response.cost for response in valid_responses)
        for response in responses:
            self.model_timings.setdefault(response.model_id, []).append(response.api_duration)
            self.model_wait_times.setdefault(response.model_id, []).append(response.duration - response.api_duration)
        return responses[:n]

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
                plt.plot(timings, label=f"{model} - Response Time", linestyle="-", linewidth=2)
                plt.plot(wait_times, label=f"{model} - Waiting Time", linestyle="--", linewidth=2)
            plt.legend()
            plt.title("Model Performance: Response and Waiting Times")
            plt.xlabel("Sample Number")
            plt.ylabel("Time (seconds)")
            plt.savefig(self.prompt_history_dir / "model_timings.png", bbox_inches="tight")
            plt.close()


async def demo():
    model_api = InferenceAPI(anthropic_num_threads=2, openai_fraction_rate_limit=1)
    anthropic_requests = [
        model_api(
            "claude-instant-1",
            "\n\nHuman: What's your name?\n\nAssistant:",
            True,
            max_tokens_to_sample=2,
        )
    ]
    oai_chat_messages = [
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
    ]
    oai_chat_models = ["gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613"]
    oai_chat_requests = [
        model_api(
            oai_chat_models,
            prompt=message,
            n=6,
            max_tokens=16_000,
            print_prompt_and_response=True,
        )
        for message in oai_chat_messages
    ]
    oai_messages = ["1 2 3", ["beforeth they cometh", "whence afterever the storm"]]
    oai_models = ["davinci-002"]
    oai_requests = [
        model_api(oai_models, prompt=message, n=1, print_prompt_and_response=True) for message in oai_messages
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
