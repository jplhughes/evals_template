from pathlib import Path
import random
import pandas as pd

from datasets import load_dataset
import fire
from tqdm import tqdm
import logging

LOGGER = logging.getLogger(__name__)

MMLU_TOPICS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def raw_to_question(raw: dict):
    question = raw["question"].rstrip("\n A:")
    base_question = f"""{question}"""
    return base_question


def incompatible_answers(label: str):
    return (
        "all of the above" in label.lower()
        or "both" in label.lower()
        or "neither of these" in label.lower()
        or "none of the above" in label.lower()
    )


def load_mmlu(
    filename: Path, topics: list[str] = MMLU_TOPICS, seed: int = 42, num_per_topic: int = 6, split: str = "validation"
):
    random.seed(seed)
    data = []
    num_skips = 0

    for topic in topics:
        dataset = load_dataset("tasksource/mmlu", topic)

        if split == "train":
            dataset = dataset["dev"]
        elif split == "validation":
            dataset = dataset["validation"]
        else:
            raise ValueError(f"Invalid split: {split}")

        LOGGER.info(f"Topic: {topic}")
        LOGGER.info(f"Dataset size: {len(dataset)}")

        # sample datapoints from set
        num_per_topic = min(num_per_topic, len(dataset))
        dataset = dataset.shuffle(seed=seed).select(range(num_per_topic))

        for source_id, item in enumerate(tqdm(dataset)):
            question = raw_to_question(item)
            correct_key = int(item["answer"])
            correct_answer = item["choices"][correct_key]

            candidates = item["choices"]
            candidates.remove(correct_answer)

            # sample negative answer from remaining choices
            negative_answer = random.choice(candidates)
            if incompatible_answers(correct_answer) or incompatible_answers(negative_answer):
                LOGGER.info(f"Skipping question with incompatible answer: {negative_answer}")
                num_skips += 1
                continue

            data.append(
                {
                    "id": source_id,
                    "question": question,
                    "correct_answer": correct_answer,
                    "negative_answer": negative_answer,
                    "topic": topic,
                }
            )

    LOGGER.info(f"Skipped {num_skips} questions")
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    fire.Fire(load_mmlu)
