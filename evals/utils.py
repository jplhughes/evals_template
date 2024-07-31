import json
import logging
import os
from pathlib import Path
import git

import yaml
from tenacity import retry, retry_if_result, stop_after_attempt

LOGGER = logging.getLogger(__name__)

LOGGING_LEVELS = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def get_repo_root() -> Path:
    """Returns repo root (relative to this file)."""
    return Path(
        git.Repo(
            __file__,
            search_parent_directories=True,
        ).git.rev_parse("--show-toplevel")
    )


def setup_environment(
    anthropic_tag: str = "ANTHROPIC_API_KEY",
    logging_level: str = "info",
    openai_tag: str = "OPENAI_API_KEY",
    organization_tag: str = "ACEDEMICNYUPEREZ_ORG",
):
    setup_logging(logging_level)
    secrets = load_secrets("SECRETS")
    os.environ["OPENAI_API_KEY"] = secrets[openai_tag]
    os.environ["OPENAI_ORG_ID"] = secrets[organization_tag]
    os.environ["ANTHROPIC_API_KEY"] = secrets[anthropic_tag]
    if "RUNPOD_API_KEY" in secrets:
        os.environ["RUNPOD_API_KEY"] = secrets["RUNPOD_API_KEY"]
    if "HF_API_KEY" in secrets:
        # https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
        os.environ["HF_TOKEN"] = secrets["HF_API_KEY"]


def setup_logging(logging_level):
    level = LOGGING_LEVELS.get(logging_level.lower(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Disable logging from openai
    logging.getLogger("openai").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.CRITICAL)


def load_secrets(file_path):
    secrets = {}
    with open(file_path) as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            secrets[key] = value
    return secrets


def load_yaml(file_path):
    with open(file_path) as f:
        content = yaml.safe_load(f)
    return content


def save_yaml(file_path, data):
    with open(file_path, "w") as f:
        yaml.dump(data, f)


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data


def save_jsonl(file_path, data):
    with open(file_path, "w") as f:
        for line in data:
            json.dump(line, f)
            f.write("\n")


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f)


@retry(
    stop=stop_after_attempt(8),
    retry=retry_if_result(lambda result: result is not True),
)
def function_with_retry(function, *args, **kwargs):
    return function(*args, **kwargs)


@retry(
    stop=stop_after_attempt(8),
    retry=retry_if_result(lambda result: result is not True),
)
async def async_function_with_retry(function, *args, **kwargs):
    return await function(*args, **kwargs)
