import logging
from pathlib import Path

import filelock

from evals.data_models import (
    EmbeddingParams,
    EmbeddingResponseBase64,
    LLMCache,
    LLMCacheModeration,
    LLMParams,
    Prompt,
    TaggedModeration,
)
from evals.data_models.hashable import deterministic_hash
from evals.utils import load_json, save_json

LOGGER = logging.getLogger(__name__)


class CacheManager:
    def __init__(self, cache_dir: Path, num_bins: int = 20):
        self.cache_dir = cache_dir
        self.num_bins = num_bins  # Number of bins for cache division
        self.in_memory_cache = {}

    @staticmethod
    def get_bin_number(hash_value, num_bins):
        # Convert the hash into an integer and find its modulo with num_bins
        return int(hash_value, 16) % num_bins

    def get_cache_file(self, prompt: Prompt, params: LLMParams) -> tuple[Path, str]:
        # Use the SHA-1 hash of the prompt for the dictionary key
        prompt_hash = prompt.model_hash()  # Assuming this gives a SHA-1 hash as a hex string
        bin_number = self.get_bin_number(prompt_hash, self.num_bins)

        # Construct the file name using the bin number
        cache_dir = self.cache_dir / params.model_hash()
        cache_file = cache_dir / f"bin{str(bin_number)}.json"

        return cache_file, prompt_hash

    def maybe_load_cache(self, prompt: Prompt, params: LLMParams):
        cache_file, prompt_hash = self.get_cache_file(prompt, params)
        if not cache_file.exists():
            return None

        if (cache_file not in self.in_memory_cache) or (prompt_hash not in self.in_memory_cache[cache_file]):
            LOGGER.info(f"Cache miss, loading from disk: {cache_file=}, {prompt_hash=}")
            with filelock.FileLock(str(cache_file) + ".lock"):
                self.in_memory_cache[cache_file] = load_json(cache_file)

        data = self.in_memory_cache[cache_file].get(prompt_hash, None)
        return None if data is None else LLMCache.model_validate_json(data)

    def save_cache(self, prompt: Prompt, params: LLMParams, responses: list):
        cache_file, prompt_hash = self.get_cache_file(prompt, params)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        new_cache_entry = LLMCache(prompt=prompt, params=params, responses=responses)

        with filelock.FileLock(str(cache_file) + ".lock"):
            cache_data = {}
            # If the cache file exists, load it; otherwise, start with an empty dict
            if cache_file.exists():
                cache_data = load_json(cache_file)

            cache_data[prompt_hash] = new_cache_entry.model_dump_json()
            save_json(cache_file, cache_data)

    def get_moderation_file(self, texts: list[str]) -> tuple[Path, str]:
        hashes = [deterministic_hash(t) for t in texts]
        hash = deterministic_hash(" ".join(hashes))
        bin_number = self.get_bin_number(hash, self.num_bins)
        if (self.cache_dir / "moderation" / f"{hash}.json").exists():
            cache_file = self.cache_dir / "moderation" / f"{hash}.json"
        else:
            cache_file = self.cache_dir / "moderation" / f"bin{str(bin_number)}.json"

        return cache_file, hash

    def maybe_load_moderation(self, texts: list[str]):
        cache_file, hash = self.get_moderation_file(texts)
        if cache_file.exists():
            all_data = load_json(cache_file)
            data = all_data.get(hash, None)
            if data is not None:
                return LLMCacheModeration.model_validate_json(data)
        return None

    def save_moderation(self, texts: list[str], moderation: list[TaggedModeration]):
        cache_file, hash = self.get_moderation_file(texts)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        new_cache_entry = LLMCacheModeration(texts=texts, moderation=moderation)

        with filelock.FileLock(str(cache_file) + ".lock"):
            cache_data = {}
            # If the cache file exists, load it; otherwise, start with an empty dict
            if cache_file.exists():
                cache_data = load_json(cache_file)

            # Update the cache data with the new responses for this prompt
            cache_data[hash] = new_cache_entry.model_dump_json()

            save_json(cache_file, cache_data)

    def get_embeddings_file(self, params: EmbeddingParams) -> tuple[Path, str]:
        hash = params.model_hash()
        bin_number = self.get_bin_number(hash, self.num_bins)

        cache_file = self.cache_dir / "embeddings" / f"bin{str(bin_number)}.json"

        return cache_file, hash

    def maybe_load_embeddings(self, params: EmbeddingParams) -> EmbeddingResponseBase64 | None:
        cache_file, hash = self.get_embeddings_file(params)
        if cache_file.exists():
            all_data = load_json(cache_file)
            data = all_data.get(hash, None)
            if data is not None:
                return EmbeddingResponseBase64.model_validate_json(data)
        return None

    def save_embeddings(self, params: EmbeddingParams, response: EmbeddingResponseBase64):
        cache_file, hash = self.get_embeddings_file(params)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with filelock.FileLock(str(cache_file) + ".lock"):
            cache_data = {}
            # If the cache file exists, load it; otherwise, start with an empty dict
            if cache_file.exists():
                cache_data = load_json(cache_file)

            # Update the cache data with the new responses for this prompt
            cache_data[hash] = response.model_dump_json()

            save_json(cache_file, cache_data)
