import asyncio
import logging
import traceback
from pathlib import Path
from string import Template

import hydra
import pandas as pd
from omegaconf import DictConfig

from evals.apis.inference.api import InferenceAPI
from evals.load.mmlu import load_mmlu
from evals.utils import async_function_with_retry, setup_environment

LOGGER = logging.getLogger(__name__)


async def get_completion(
    index: int, prompt: list[dict[str, str]], language_model: DictConfig, inference_api: InferenceAPI
) -> dict[str, str]:
    try:
        responses = await inference_api(
            model_ids=language_model.model,
            prompt=prompt,
            temperature=language_model.temperature,
            max_tokens=language_model.max_tokens,
            top_p=language_model.top_p,
            num_candidates_per_completion=language_model.num_candidates_per_completion,
            insufficient_valids_behaviour=language_model.insufficient_valids_behaviour,
            is_valid=lambda x: "Answer:" in x,
        )
        answer = responses[0].completion
        complete = True
        inference_api.log_model_timings()
        LOGGER.info(f"Completed row {index}\tRunning cost: ${inference_api.running_cost:.3f}")
    except RuntimeError as e:
        complete = False
        answer = traceback.format_exc()
        LOGGER.warning(f"Failed row {index} with error {e}")
        LOGGER.warning(answer)
    return {
        "answer": answer,
        "complete": complete,
    }


def process_prompt(prompt: DictConfig, swap: bool, row: pd.Series) -> list[dict[str, str]]:
    answer_a = row["correct_answer"] if not swap else row["negative_answer"]
    answer_b = row["negative_answer"] if not swap else row["correct_answer"]

    messages = []
    for message in prompt.messages:
        t = Template(message["content"])
        messages.append(
            {
                "role": message["role"],
                "content": t.safe_substitute(question=row["question"], answer_a=answer_a, answer_b=answer_b),
            }
        )
    return messages


async def run_dataset(cfg: DictConfig, inference_api: InferenceAPI, filename: Path) -> bool:
    # load dataset and filter out completed rows
    full_df = pd.read_csv(filename)
    if cfg.limit is not None:
        full_df = full_df.head(cfg.limit)
    if "answer" not in full_df.columns:
        full_df["answer"] = ""
    if "complete" not in full_df.columns:
        full_df["complete"] = False
    df = full_df[~(full_df["complete"])]

    # run each question concurrently
    LOGGER.info(f"Processing {len(df)} rows")
    prompts = [process_prompt(cfg.prompt, cfg.swap, row) for _, row in df.iterrows()]
    tasks = [get_completion(i, prompt, cfg.language_model, inference_api) for i, prompt in enumerate(prompts)]
    results = await asyncio.gather(*tasks)

    # update dataframe with results
    completed = sum([bool(result["complete"]) for result in results])
    LOGGER.info(f"Processed {len(results)} rows. {completed} were complete.")
    df.update(
        pd.DataFrame(
            {
                "answer": [result["answer"] for result in results],
                "complete": [result["complete"] for result in results],
            },
            index=df.index,
        )
    )
    full_df.update(df)
    full_df.to_csv(filename, index=False, encoding="utf-8")

    # return whether all rows are complete
    if full_df["complete"].eq(True).all():
        LOGGER.info("All rows complete!")
        return True
    else:
        LOGGER.info("Not all rows complete. Retrying...")
        return False


async def async_main(cfg: DictConfig):
    LOGGER.info(f"Using experiment directory {cfg.exp_dir}")
    LOGGER.info(f"Using model {cfg.language_model.model}")
    LOGGER.info(f"Using method {cfg.prompt.method}")
    LOGGER.info(f"Using swap: {cfg.swap}")

    # setup api handler
    setup_environment(anthropic_tag=cfg.anthropic_tag, logging_level=cfg.logging)
    inference_api = InferenceAPI(
        anthropic_num_threads=cfg.anthropic_num_threads,
        openai_fraction_rate_limit=cfg.openai_fraction_rate_limit,
        organization=cfg.organization,
        print_prompt_and_response=cfg.print_prompt_and_response,
        exp_dir=Path(cfg.exp_dir),
    )

    # load dataset and save to file
    exp_dir = Path(cfg.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    filename = exp_dir / f"data{cfg.seed}_swap{cfg.swap}.csv"
    if not filename.exists():
        LOGGER.info(f"File {filename} does not exist. Creating...")
        load_mmlu(filename, topics=["high_school_mathematics"], num_per_topic=25)

    # run dataset (with retry)
    complete = await async_function_with_retry(
        run_dataset,
        cfg,
        inference_api,
        filename,
    )
    return complete


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
