import io
import json
from pathlib import Path

import openai
import pandas as pd
import fire

from evals.utils import save_jsonl, setup_environment


def list_finetunes(limit: int = 100, organization: str = "ACEDEMICNYUPEREZ_ORG") -> None:
    setup_environment(organization=organization)
    finetunes = openai.FineTuningJob.list(limit=limit)
    print(finetunes)


def delete_file(file_id: str, organization: str = "ACEDEMICNYUPEREZ_ORG") -> None:
    setup_environment(organization=organization)
    openai.File.delete(file_id)
    print(f"Deleted file {file_id}")


def delete_all_files(organization: str = "ACEDEMICNYUPEREZ_ORG") -> None:
    setup_environment(organization=organization)
    files = openai.File.list().data  # type: ignore
    for file in files:
        try:
            if "results" not in file["purpose"]:
                delete_file(file["id"], organization)
        except Exception as e:
            print(f"Failed to delete file {file['id']} with error {e}")
    print("deleted all files")


def list_all_files(organization: str = "ACEDEMICNYUPEREZ_ORG") -> None:
    setup_environment(organization=organization)
    files = openai.File.list().data  # type: ignore
    print(files)


def cancel_job(job_id: str, organization: str = "ACEDEMICNYUPEREZ_ORG") -> None:
    setup_environment(organization=organization)
    print(openai.FineTuningJob.cancel(job_id))


def retrieve_job(job_id: str, organization: str = "ACEDEMICNYUPEREZ_ORG") -> None:
    setup_environment(organization=organization)
    print(openai.FineTuningJob.retrieve(job_id))


def download_result_file(result_file_id: str, organization: str = "ACEDEMICNYUPEREZ_ORG") -> None:
    setup_environment(organization=organization)
    file = openai.File.retrieve(result_file_id)
    downloaded: bytes = openai.File.download(result_file_id)
    # use pandas
    csv = pd.read_csv(io.BytesIO(downloaded))
    for k, v in file.items():
        print(f"{k}: {v}")
    print(csv.to_markdown())


def download_training_file(training_file_id: str, organization: str = "ACEDEMICNYUPEREZ_ORG") -> None:
    setup_environment(organization=organization)
    openai.File.retrieve(training_file_id)
    downloaded: bytes = openai.File.download(training_file_id)
    # these are jsonl files, so its a list of dicts
    output = [json.loads(line) for line in downloaded.decode().split("\n") if line]
    print(len(output))


def save_test_file(file_path: Path = Path.home() / "test.jsonl"):
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "bye"},
    ]
    data = [{"messages": messages} for _ in range(10)]
    save_jsonl(file_path, data)


if __name__ == "__main__":
    fire.Fire()
