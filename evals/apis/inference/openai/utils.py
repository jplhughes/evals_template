import tiktoken

COMPLETION_MODELS = {
    "davinci-002",
    "babbage-002",
    "text-davinci-003",
    "text-davinci-002",
    "gpt-4-base",
    "gpt-3.5-turbo-instruct",
}

EMBEDDING_MODELS = (
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
)

_GPT_4_MODELS = (
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
)
_GPT_TURBO_MODELS = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
)
GPT_CHAT_MODELS = set(_GPT_4_MODELS + _GPT_TURBO_MODELS)


def count_tokens(text: str) -> int:
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def get_max_context_length(model_id: str) -> int:
    match model_id:
        case (
            "gpt-4o"
            | "gpt-4o-2024-05-13"
            | "gpt-4-turbo"
            | "gpt-4-turbo-2024-04-09"
            | "gpt-4-0125-preview"
            | "gpt-4-turbo-preview"
            | "gpt-4-1106-preview"
            | "gpt-4-vision-preview"
        ):
            return 128_000

        case "gpt-4" | "gpt-4-0613":
            return 8192

        case "gpt-4-32k" | "gpt-4-32k-0613":
            return 32_768

        case "gpt-3.5-turbo" | "gpt-3.5-turbo-0613":
            return 4096

        case "gpt-3.5-turbo-1106" | "gpt-3.5-turbo-0125" | "gpt-3.5-turbo-16k" | "gpt-3.5-turbo-16k-0613":
            return 16_385

        case "gpt-3.5-turbo-instruct":
            return 4096

        case _:
            raise ValueError(f"Invalid model id: {model_id}")


def get_rate_limit(model_id: str) -> tuple[int, int]:
    """
    Returns the (tokens per min, request per min) for the given model id.
    """
    if "ft:gpt-3.5-turbo" in model_id:
        return 2_000_000, 10_000
    match model_id:
        case "gpt-4o" | "gpt-4o-2024-05-13":
            return 5_000_000, 10_000
        case (
            "gpt-4-turbo"
            | "gpt-4-turbo-2024-04-09"
            | "gpt-4-0125-preview"
            | "gpt-4-turbo-preview"
            | "gpt-4-1106-preview"
        ):
            return 2_000_000, 10_000
        case "gpt-4" | "gpt-4-0613" | "gpt-4-32k" | "gpt-4-32k-0613":
            return 300_000, 10_000
        case (
            "gpt-3.5-turbo"
            | "gpt-3.5-turbo-0613"
            | "gpt-3.5-turbo-1106"
            | "gpt-3.5-turbo-0125"
            | "gpt-3.5-turbo-16k"
            | "gpt-3.5-turbo-16k-0613"
            | "gpt-3.5-turbo-instruct"
        ):
            return 2_000_000, 10_000
        case _:
            raise ValueError(f"Invalid model id: {model_id}")


def price_per_token(model_id: str) -> tuple[float, float]:
    """
    Returns the (input token, output token) price for the given model id.
    """

    if model_id in (
        "gpt-4o",
        "gpt-4o-2024-05-13",
    ):
        prices = 0.005, 0.015
    elif model_id in (
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-1106-preview",
        "gpt-4-0125-preview",
        "gpt-4-turbo-preview",
    ):
        prices = 0.01, 0.03
    elif model_id == "gpt-3.5-turbo-0125":
        prices = 0.0005, 0.0015
    elif model_id == "gpt-3.5-turbo-1106":
        prices = 0.001, 0.002
    elif model_id.startswith("gpt-4"):
        prices = 0.03, 0.06
    elif model_id.startswith("gpt-4-32k"):
        prices = 0.06, 0.12
    elif model_id.startswith("gpt-3.5-turbo-16k"):
        prices = 0.003, 0.004
    elif model_id.startswith("gpt-3.5-turbo"):
        prices = 0.0015, 0.002
    elif model_id == "davinci-002":
        prices = 0.002, 0.002
    elif model_id == "babbage-002":
        prices = 0.0004, 0.0004
    elif model_id == "text-davinci-003" or model_id == "text-davinci-002":
        prices = 0.02, 0.02
    elif "ft:gpt-3.5-turbo" in model_id:
        prices = 0.003, 0.006
    elif model_id == "text-embedding-3-small":
        prices = 0.00002, 0.00002
    elif model_id == "text-embedding-3-large":
        prices = 0.00013, 0.00013
    elif model_id == "text-embedding-ada-002":
        prices = 0.00010, 0.00010
    else:
        raise ValueError(f"Invalid model id: {model_id}")

    return tuple(price / 1000 for price in prices)


def finetune_price_per_token(model_id: str) -> float | None:
    """
    Returns price per 1000 tokens.
    Returns None if the model does not support fine-tuning.

    See https://openai.com/pricing and
    https://platform.openai.com/docs/guides/fine-tuning/what-models-can-be-fine-tuned
    for details.
    """
    match model_id:
        case "gpt-3.5-turbo-1106" | "gpt-3.5-turbo-0613" | "gpt-3.5-turbo":
            return 0.008
        case "gpt-4-0613":
            raise NotImplementedError  # TODO: Add price
        case "davinci-002":
            return 0.006
        case "babbage-002":
            return 0.0004
        case _:
            return None


def get_equivalent_model_ids(model_id: str) -> tuple[str, ...]:
    """
    Updated 2024-04-25 by Tony. This should be periodically updated.
    """

    # https://platform.openai.com/docs/models/gpt-3-5
    gpt_3_5_turbo_models = ("gpt-3.5-turbo", "gpt-3.5-turbo-0125")
    if model_id in gpt_3_5_turbo_models:
        return gpt_3_5_turbo_models
    gpt_3_5_turbo_16k_models = ("gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613")
    if model_id in gpt_3_5_turbo_16k_models:
        return gpt_3_5_turbo_16k_models

    # https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
    gpt_4_models = ("gpt-4", "gpt-4-0613")
    if model_id in gpt_4_models:
        return gpt_4_models
    gpt_4_32k_models = ("gpt-4-32k", "gpt-4-32k-0613")
    if model_id in gpt_4_32k_models:
        return gpt_4_32k_models
    # gpt4t_models = ("gpt-4-turbo-preview", "gpt-4-0125-preview")
    # if model_id in gpt4t_models:
    #     return gpt4t_models

    return (model_id,)
