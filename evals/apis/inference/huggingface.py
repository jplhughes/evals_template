import asyncio
import collections
import logging
import time
from pathlib import Path
from traceback import format_exc

import aiohttp

from evals.data_models import LLMResponse, Prompt, StopReason
from evals import math_utils

from .model import InferenceAPIModel

HUGGINGFACE_MODELS = {
    # robust-rlhf endpoints
    "meta-llama/Meta-Llama-Guard-2-8B": "https://viu7xi0u1yvsw1i1.us-east-1.aws.endpoints.huggingface.cloud",
    "cais/HarmBench-Llama-2-13b-cls": "https://bs4v12dq6qct1vnl.us-east-1.aws.endpoints.huggingface.cloud",
    "cais/HarmBench-Mistral-7b-val-cls": "https://zxgc23txss3vps53.us-east-1.aws.endpoints.huggingface.cloud",
    "cais/zephyr_7b_r2d2": "https://kfxoz77lc3ub5jmj.us-east-1.aws.endpoints.huggingface.cloud",
    "HuggingFaceH4/zephyr-7b-beta": "https://vzru5d0424di3uq9.us-east-1.aws.endpoints.huggingface.cloud",
    # John's endpoints
    "LlamaGuard-7b": "https://gtja605syok1hr6f.us-east-1.aws.endpoints.huggingface.cloud",
    # "LlamaGuard2-8b": "https://dueh6zo0ofo1yulw.us-east-1.aws.endpoints.huggingface.cloud", # use robust-rlhf endpoint
    "Llama3-8b": "https://oh0f3nw78jyonmui.us-east-1.aws.endpoints.huggingface.cloud",
    "Llama2-7b": "https://mewd3xl9sze079tl.us-east-1.aws.endpoints.huggingface.cloud",
    "Llama2-13b": "https://wg9d0r9dlb1fxn5b.us-east-1.aws.endpoints.huggingface.cloud",
    "Llama2-70b": "https://q612yteumkfz85d3.us-east-1.aws.endpoints.huggingface.cloud",
    # "zephyr-7b-r2d2": "https://unb5o4ngjb6rs11m.us-east-1.aws.endpoints.huggingface.cloud", # use robust-rlhf endpoint
    # "zephyr-7b-beta": "https://yqajs6gbkll3nego.us-east-1.aws.endpoints.huggingface.cloud", # use robust-rlhf endpoint
}

LOGGER = logging.getLogger(__name__)


class HuggingFaceModel(InferenceAPIModel):
    def __init__(
        self,
        num_threads: int,
        token: str,
        prompt_history_dir: Path | None = None,
    ):
        self.num_threads = num_threads
        self.prompt_history_dir = prompt_history_dir
        self.available_requests = asyncio.BoundedSemaphore(int(self.num_threads))

        self.token = token
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        self.kwarg_change_name = {
            "temperature": "temperature",
            "max_tokens": "max_new_tokens",
            "top_p": "top_p",
            "logprobs": "top_n_tokens",
            "seed": "seed",
        }

        self.stop_reason_map = {
            "length": StopReason.MAX_TOKENS.value,
            "eos_token": StopReason.STOP_SEQUENCE.value,
        }

    async def query(self, model_url, payload, session):
        async with session.post(model_url, headers=self.headers, json=payload) as response:
            return await response.json()

    async def run_query(self, model_url, input_text, params):
        payload = {
            "inputs": input_text,
            "parameters": params | {"details": True, "return_full_text": False},
        }
        async with aiohttp.ClientSession() as session:
            response = await self.query(model_url, payload, session)
            if "error" in response:
                raise RuntimeError(f"API Error: {response['error']}")
            return response

    @staticmethod
    def parse_logprobs(response_details) -> list[dict[str, float]] | None:
        if "top_tokens" not in response_details:
            return None

        logprobs = []
        for logprobs_at_pos in response_details["top_tokens"]:
            possibly_duplicated_logprobs = collections.defaultdict(list)
            for token_info in logprobs_at_pos:
                possibly_duplicated_logprobs[token_info["text"]].append(token_info["logprob"])

            logprobs.append({k: math_utils.logsumexp(vs) for k, vs in possibly_duplicated_logprobs.items()})

        return logprobs

    async def __call__(
        self,
        model_ids: tuple[str, ...],
        prompt: Prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        is_valid=lambda x: True,
        **kwargs,
    ) -> list[LLMResponse]:
        assert self.token is not None and self.token.startswith("hf_"), f"Invalid token {self.token}"

        start = time.time()
        assert len(model_ids) == 1, "Implementation only supports one model at a time."

        (model_id,) = model_ids
        assert model_id in HUGGINGFACE_MODELS, f"Model {model_id} not supported"
        model_url = HUGGINGFACE_MODELS[model_id]
        prompt_file = self.create_prompt_history_file(prompt, model_id, self.prompt_history_dir)

        # un-comment to get some context length issue inputs through
        # if "Llama2" in model_id:
        #     if "max_tokens" in kwargs:
        #         kwargs["max_tokens"] = min(kwargs["max_tokens"], 800)

        LOGGER.debug(f"Making {model_id} call")
        response_data = None
        duration = None
        api_duration = None
        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    api_start = time.time()
                    params = {self.kwarg_change_name[k]: v for k, v in kwargs.items() if k in self.kwarg_change_name}
                    if "temperature" in kwargs:
                        if kwargs["temperature"] == 0:
                            # Huggingface API doesn't support temperature = 0
                            # Need to set do_sample = False instead
                            del params["temperature"]
                            params["do_sample"] = False
                        else:
                            params["do_sample"] = True
                    else:
                        params["do_sample"] = True
                        params["temperature"] = 1

                    response_data = await self.run_query(
                        model_url,
                        prompt.hf_format(hf_model_id=model_id),
                        params,
                    )
                    api_duration = time.time() - api_start
                    if not is_valid(response_data[0]["generated_text"]):
                        raise RuntimeError(f"Invalid response according to is_valid {response_data}")
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"

                if ("API Error: Input validation error: `inputs` must have less than" in str(e)) or (
                    "API Error: Input validation error: `inputs` tokens + `max_new_tokens` must be <=" in str(e)
                ):
                    # Early exit on context length errors.
                    raise e

                if "503 Service Unavailable" in str(e):
                    LOGGER.warn(f"503 Service Unavailable error. Waiting 60 seconds before retrying. (Attempt {i})")
                    await asyncio.sleep(60)
                else:
                    LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                    await asyncio.sleep(1.5**i)
            else:
                break

        if response_data is None:
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")

        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        assert len(response_data) == 1, f"Expected length 1, got {len(response_data)}"

        response = LLMResponse(
            model_id=model_id,
            completion=response_data[0]["generated_text"],
            stop_reason=self.stop_reason_map[response_data[0]["details"]["finish_reason"]],
            duration=duration,
            api_duration=api_duration,
            logprobs=self.parse_logprobs(response_details=response_data[0]["details"]),
            cost=0,
        )
        responses = [response]

        self.add_response_to_prompt_file(prompt_file, responses)
        if print_prompt_and_response:
            prompt.pretty_print(responses)

        return responses
