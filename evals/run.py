import asyncio
import logging
import traceback
from pathlib import Path
from string import Template

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from evals.apis.inference.api import InferenceAPI
from evals.data_models.language_model import LLMParams
from evals.data_models.messages import ChatMessage, PromptTemplate, Prompt
from evals.load.mmlu import load_mmlu
from evals.utils import async_function_with_retry, setup_environment

LOGGER = logging.getLogger(__name__)


class DatasetRunner:
    def __init__(self, prompt_template: PromptTemplate, llm_params: LLMParams, inference_api: InferenceAPI, swap: bool):
        self.prompt_template = prompt_template
        self.llm_params = llm_params
        self.inference_api = inference_api
        self.swap = swap

    async def run(self, index: int, row: pd.Series) -> dict:
        prompt = self.process_prompt(row)
        try:
            responses = await self.inference_api(
                model_ids=self.llm_params.model,
                prompt=prompt,
                temperature=self.llm_params.temperature,
                max_tokens=self.llm_params.max_tokens,
                top_p=self.llm_params.top_p,
                print_prompt_and_response=self.llm_params.print_prompt_and_response,
                num_candidates_per_completion=self.llm_params.num_candidates_per_completion,
                insufficient_valids_behaviour=self.llm_params.insufficient_valids_behaviour,
                is_valid=lambda x: "Answer:" in x,
            )
            answer = responses[0].completion
            complete = True
            self.inference_api.log_model_timings()
            LOGGER.info(f"Completed row {index}\tRunning cost: ${self.inference_api.running_cost:.3f}")
        except RuntimeError as e:
            complete = False
            answer = traceback.format_exc()
            LOGGER.warning(f"Failed row {index} with error {e}")
            LOGGER.warning(answer)
        return {
            "answer": answer,
            "complete": complete,
        }

    def process_prompt(self, row: pd.Series) -> Prompt:
        answer_a = row["correct_answer"] if not self.swap else row["negative_answer"]
        answer_b = row["negative_answer"] if not self.swap else row["correct_answer"]

        messages = []
        for message in self.prompt_template.messages:
            t = Template(message.content)
            content = t.safe_substitute(question=row["question"], answer_a=answer_a, answer_b=answer_b)
            messages.append(ChatMessage(role=message.role, content=content))

        return Prompt(messages=messages)


async def run_dataset(filename: str, dataset_runner: DatasetRunner, limit: int = None) -> bool:
    # load dataset and filter out completed rows
    full_df = pd.read_csv(filename)
    if limit is not None:
        full_df = full_df.head(limit)
    if "answer" not in full_df.columns:
        full_df["answer"] = ""
    if "complete" not in full_df.columns:
        full_df["complete"] = False
    if "swap" not in full_df.columns:
        full_df["swap"] = ""
    df = full_df[~(full_df["complete"])]

    # run each question concurrently
    LOGGER.info(f"Processing {len(df)} rows")
    tasks = [dataset_runner.run(i, row) for i, row in df.iterrows()]
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
        exp_dir=Path(cfg.exp_dir),
    )
    # load configs
    prompt_parts = PromptTemplate(**OmegaConf.to_container(cfg.prompt, resolve=True))
    llm_params = LLMParams(**OmegaConf.to_container(cfg.language_model, resolve=True))
    dataset_runner = DatasetRunner(prompt_parts, llm_params, inference_api, cfg.swap)

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
        filename,
        dataset_runner,
        limit=cfg.limit,
    )
    return complete


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
