import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import openai
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


def setup_environment(
    anthropic_tag: str = "ANTHROPIC_API_KEY",
    logging_level: str = "info",
    openai_tag: str = "OPENAI_API_KEY",
):
    setup_logging(logging_level)
    secrets = load_secrets("SECRETS")
    openai.api_key = secrets[openai_tag]
    os.environ["ANTHROPIC_API_KEY"] = secrets[anthropic_tag]


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


@retry(
    stop=stop_after_attempt(16),
    retry=retry_if_result(lambda result: result is not True),
)
def function_with_retry(function, *args, **kwargs):
    return function(*args, **kwargs)


@retry(
    stop=stop_after_attempt(16),
    retry=retry_if_result(lambda result: result is not True),
)
async def async_function_with_retry(function, *args, **kwargs):
    return await function(*args, **kwargs)


def log_model_timings(api_handler, save_location="./model_timings.png"):
    if len(api_handler.model_timings) > 0:
        plt.figure(figsize=(10, 6))
        for model in api_handler.model_timings:
            timings = np.array(api_handler.model_timings[model])
            wait_times = np.array(api_handler.model_wait_times[model])
            LOGGER.info(
                f"{model}: response {timings.mean():.3f}, waiting {wait_times.mean():.3f} (max {wait_times.max():.3f}, min {wait_times.min():.3f})"
            )
            plt.plot(timings, label=f"{model} - Response Time", linestyle="-", linewidth=2)
            plt.plot(wait_times, label=f"{model} - Waiting Time", linestyle="--", linewidth=2)
        plt.legend()
        plt.title("Model Performance: Response and Waiting Times")
        plt.xlabel("Sample Number")
        plt.ylabel("Time (seconds)")
        plt.savefig(save_location, dpi=300)
        plt.close()
