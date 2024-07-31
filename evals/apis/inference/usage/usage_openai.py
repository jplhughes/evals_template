import dataclasses
import logging
import os

import requests
import simple_parsing

from evals.utils import setup_environment, load_secrets

logger = logging.getLogger(__name__)


def extract_usage(response):
    requests_left = float(response.headers["x-ratelimit-remaining-requests"])
    requests_limit = float(response.headers["x-ratelimit-limit-requests"])
    request_usage = 1 - (requests_left / requests_limit)
    tokens_left = float(response.headers["x-ratelimit-remaining-tokens"])
    tokens_limit = float(response.headers["x-ratelimit-limit-tokens"])
    token_usage = 1 - (tokens_left / tokens_limit)
    overall_usage = max(request_usage, token_usage)
    return overall_usage


def get_ratelimit_usage(data, org_id, endpoint):
    try:
        api_key = os.environ["OPENAI_API_KEY"]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Organization": org_id,
        }
        response = requests.post(
            endpoint,
            headers=headers,
            json=data,
            timeout=20,
        )

        return extract_usage(response)
    except Exception as e:
        logger.warning(f"Error fetching ratelimit usage: {e}")
        return -1


def fetch_ratelimit_usage(org_id, model_name) -> float:
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say 1"}],
    }
    return get_ratelimit_usage(data, org_id, "https://api.openai.com/v1/chat/completions")


def fetch_ratelimit_usage_base(org_id, model_name) -> float:
    data = {"model": model_name, "prompt": "a", "max_tokens": 1}
    return get_ratelimit_usage(data, org_id, "https://api.openai.com/v1/completions")


def get_current_openai_model_usage(models, organizations) -> None:
    secrets = load_secrets("SECRETS")
    print("\nModel usage: 1 is hitting rate limits, 0 is not in use. -1 is error.\n")
    for organization in organizations:
        print(f"{organization}:")
        org_id = secrets[organization]
        for model_name in models:
            usage = fetch_ratelimit_usage(org_id, model_name)
            print(f"\t{model_name}:\t{usage:.2f}")
        print()


@dataclasses.dataclass
class Config:
    models: list[str] = dataclasses.field(
        default_factory=lambda: [
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
            "gpt-4-1106-preview",
        ]
    )
    organizations: list[str] = dataclasses.field(default_factory=lambda: ["ACEDEMICNYUPEREZ_ORG", "FARAI_ORG"])


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Config, dest="experiment_config")
    args = parser.parse_args()
    cfg: Config = args.experiment_config

    setup_environment()
    get_current_openai_model_usage(cfg.models, cfg.organizations)
