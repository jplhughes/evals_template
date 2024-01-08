from pathlib import Path

from evals.data_models.inference import LLMParams, LLMResponse
from evals.data_models.cache import LLMCache
from evals.data_models.messages import Prompt

from evals.utils import load_json, save_json


class CacheManager:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    def get_cache_file(self, prompt: Prompt, params: LLMParams) -> Path:
        return self.cache_dir / params.model_hash() / f"{prompt.model_hash()}.json"

    def maybe_load_cache(self, prompt: Prompt, params: LLMParams):
        cache_file = self.get_cache_file(prompt, params)
        if cache_file.exists():
            data = load_json(cache_file)
            return LLMCache.model_validate_json(data)
        return None

    def save_cache(self, prompt: Prompt, params: LLMParams, responses: list[LLMResponse]):
        cache_file = self.get_cache_file(prompt, params)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        cache = LLMCache(prompt=prompt, params=params, responses=responses)
        data = cache.model_dump_json(indent=2)
        save_json(cache_file, data)
