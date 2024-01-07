import datetime
import logging
import time
from pathlib import Path
from typing import Any, Optional

import openai
import typer
from openai.error import APIConnectionError, RateLimitError
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from evals.finetuning.syncer import WandbSyncer
from evals.utils import load_jsonl, setup_environment

logger = logging.getLogger(__name__)


class FineTuneHyperParams(BaseModel):
    n_epochs: int = 1
    # TODO: the new api doesn't have these params?
    # batch_size: int = 1
    # learning_rate_multiplier: float = 1.0


class FineTuneParams(BaseModel):
    model: str
    suffix: str = None
    hyperparameters: FineTuneHyperParams


class FinetuneJob(BaseModel):
    model: str
    id: str  # job id


class FinetunedJobResults(BaseModel):
    fine_tuned_model: str
    result_files: list[str] = []
    trained_tokens: int


@retry(
    retry=retry_if_exception_type(APIConnectionError),
    stop=stop_after_attempt(8),
    wait=wait_fixed(30),
)
def wait_until_uploaded_file_id_is_ready(file_id: str) -> None:
    while True:
        file = openai.File.retrieve(file_id)
        if file["status"] == "processed":
            return
        time.sleep(1)


def wait_until_finetune_job_is_ready(finetune_job_id: str) -> FinetunedJobResults:
    """Returns the fine tuned model id"""
    while True:
        finetune_job = openai.FineTuningJob.retrieve(finetune_job_id)
        if finetune_job["status"] == "succeeded":
            print(finetune_job)
            return FinetunedJobResults.parse_obj(finetune_job)
        time.sleep(1)


def confirm_to_continue(file_path: Path) -> None:
    # nice string like /home/.../file.jsonl
    file_path_str = file_path.absolute().as_posix()
    print(f"About to upload {file_path_str}. Continue? (y/n)")
    response = input()
    while response not in ["y", "n"]:
        print(f"Please enter y or n. You entered {response}")
        response = input()
    if response == "n":
        exit(0)
    print("Continuing with upload")
    return None


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    stop=stop_after_attempt(8),
    wait=wait_fixed(30),
)
def queue_finetune(
    file_id: str,
    model: str,
    hyperparameters: FineTuneHyperParams,
    suffix: str = None,
    val_file_id: str = None,
) -> FinetuneJob:
    # Keep retrying until we can queue the finetune job
    finetune_job_resp = openai.FineTuningJob.create(
        training_file=file_id,
        model=model,
        hyperparameters=hyperparameters.dict(),
        suffix=suffix,
        validation_file=val_file_id,
    )

    print(f"Started finetune job. {finetune_job_resp}")
    parsed_job_resp: FinetuneJob = FinetuneJob.parse_obj(finetune_job_resp)
    return parsed_job_resp


def upload_file(data_path: Path, params: FineTuneParams):
    now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name = f"{params.model}-{now_time}_{data_path.name}"
    file_upload_resp: dict[str, Any] = openai.File.create(  # type: ignore[reportGeneralTypeIssues]
        file=open(data_path, "rb"),
        purpose="fine-tune",
        user_provided_filename=file_name,
    )
    file_id = file_upload_resp["id"]
    print(f"Starting file upload. {file_id}\n{file_name}")
    wait_until_uploaded_file_id_is_ready(file_id=file_id)
    print(f"Uploaded file to openai. {file_upload_resp}\n{file_name}")
    return file_id


def run_finetune(
    params: FineTuneParams,
    data_path: Path,
    syncer: Optional[WandbSyncer] = None,
    ask_to_validate_training: bool = True,
    val_data_path: Optional[Path] = None,
) -> str:
    """
    Pass syncer=None to disable wandb logging
    """
    samples = load_jsonl(data_path)
    if ask_to_validate_training:
        confirm_to_continue(data_path)
    if syncer:
        syncer.update_parameters(params=params.dict())
        syncer.upload_training_file(data_path)
        syncer.update_n_samples(n_samples=len(samples))

    file_id = upload_file(data_path=data_path, params=params)
    if syncer:
        syncer.update_openai_file_id(openai_file_id=file_id)

    if val_data_path:
        val_file_id = upload_file(data_path=val_data_path, params=params)
    else:
        val_file_id = None
    finetune_job_resp = queue_finetune(
        file_id=file_id,
        model=params.model,
        hyperparameters=params.hyperparameters,
        suffix=params.suffix,
        val_file_id=val_file_id,
    )
    print(f"Started finetune job. {finetune_job_resp}")

    if syncer:
        syncer.update_finetune_job_id(finetune_job_id=finetune_job_resp.id)
    result: FinetunedJobResults = wait_until_finetune_job_is_ready(finetune_job_id=finetune_job_resp.id)
    model_id = result.fine_tuned_model
    print(f"Fine tuned model id: {model_id}. You can now use this model in the API")
    if syncer:
        syncer.update_finetune_model_id(finetune_model_id=model_id)
        syncer.update_training_results(results_id=result.result_files[0])
        syncer.end()
    return model_id


def main(
    data_path: Path,
    model: str = "gpt-3.5-turbo",
    n_epochs: int = 1,
    project_name: str = "seri-ous",
    val_data_path: Optional[Path] = None,
    notes: Optional[str] = None,
    more_config: str = None,
    ask_to_validate_training: bool = True,
    organization: str = "ACEDEMICNYUPEREZ_ORG",
    logger_level: str = "info",
    use_wandb: bool = True,
) -> str:
    assert " " not in notes, "Notes cannot have spaces, use underscores instead"
    setup_environment(organization=organization, logger_level=logger_level)
    params = FineTuneParams(
        model=model,
        hyperparameters=FineTuneHyperParams(n_epochs=n_epochs),
        suffix=notes,
    )
    if use_wandb:
        syncer = WandbSyncer.create(project_name=project_name, notes=notes)
        if more_config:
            more_config = {k: v for k, v in [x.split("=") for x in more_config.split(",")]}
            syncer.update_parameters_with_dict(params=more_config)
    else:
        syncer = None

    return run_finetune(
        params=params,
        data_path=data_path,
        syncer=syncer,
        ask_to_validate_training=ask_to_validate_training,
        val_data_path=val_data_path,
    )


if __name__ == "__main__":
    typer.run(main)
