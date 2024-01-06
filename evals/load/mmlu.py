import argparse
import csv
import random

from datasets import load_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--filename", help="filename of output csv")
args = parser.parse_args()

SEED = 42
random.seed(SEED)
FILENAME = args.filename
NUM_PER_CONFIG = 6

# Load dataset
configs = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    # "business_ethics",
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


cot_prompt = """Here are a set of example questions and answers:
Q. Which of the following is considered an acid anhydride?
A) SO2
B) HCl

Thinking step by step, we can see that SO2 is a nonmetal oxide, and HCl is a nonmetal halide.
Nonmetal oxides are considered acid anhydrides, so the correct answer is A.
Answer: A


Q. A fertilized ovum is also known as
A) an embryo.
B) a zygote.

Thinking step by step, am embryo is a multicellular organism in its early stages of development.
When a sperm cell fertilizes an egg cell, the resulting cell is called a zygote.
An ovum is an egg cell.
Thus the correct answer is B.
Answer: B


Q. Judge took judicial notice of some facts at the beginning of the trial. Which of the following is not an appropriate kind of fact for judicial notice?
A) Facts that have been asserted by individual political organizations.
B) Indisputable facts.

Thinking step by step, facts that have been asserted by individual political organizations may be disputed.
Disputed facts are not appropriate for judicial notice.
Thus the correct answer is A.
Answer: A


Q. A new smartwatch is manufactured in one part of a factory, then secured for shipping in another, independent part of the factory. The weight of the smartwatch has a mean of 62 grams and a standard deviation of 1.0 grams. The weight of the packaging (box, user's guide, bubble wrap, etc.) has a mean of 456 grams and a standard deviation of 6 grams. Together, the distribution of the weight of the smartwatch and its packaging would have the following mean and standard deviation
A) Mean 518 grams; standard deviation 6.1 grams
B) Mean 518 grams; standard deviation 7.0 grams

Thinking step by step, the weight of the smartwatch is 62 grams, and the weight of the packaging is 456 grams.
The mean of the distribution of the weight of the smartwatch and its packaging is 62 + 456 = 518 grams.
To calculate the standard deviation of the distribution of the weight of the smartwatch and its packaging, we need to add the variances of the two distributions.
The variance of the distribution of the weight of the smartwatch is 1.0^2 = 1.0.
The variance of the distribution of the weight of the packaging is 6^2 = 36.
The variance of the distribution of the weight of the smartwatch and its packaging is 1.0^2 + 36^2 = 37.
The standard deviation of the distribution of the weight of the smartwatch and its packaging is the square root of 37 grams^2, which is 6.1 grams.
Thus the correct answer is A.


Which expression is equivalent to 5 x 9?
A) (5 x 5) + (5 x 4)
B) (5 x 4) x (6 x 5)

Thinking step by step,  5 x 9 = 45.
(5 x 5) + (5 x 4) = 25 + 20 = 45.
Thus the correct answer is A.
Answer: A

"""
num_skips = 0
with open(FILENAME, "w", newline="", encoding="utf-8") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(
        [
            "id",
            "question",
            "prompt",
            "correct answer",
            "negative answer",
            "complete",
            "transcript",
            "answer",
            "cot prompt",
            "topic",
        ]
    )

    for config in configs:
        dataset = load_dataset("tasksource/mmlu", config)
        n_shot = 5
        prompt_idx = 0

        train_dataset = dataset["dev"]
        val_dataset = dataset["validation"]
        # test_dataset = dataset["test"]

        print(f"Config: {config}")
        print("Train size: ", len(train_dataset["question"]))
        print("Val size: ", len(val_dataset["choices"]))

        # sample 5 datapoints from validation set
        val_dataset = val_dataset.shuffle(seed=SEED).select(range(NUM_PER_CONFIG))
        # print("Test size: ", len(test_dataset["context"]))

        for source_id, item in enumerate(tqdm(val_dataset)):
            question = "Can you answer the following question as accurately as possible?\n"
            question += raw_to_question(item)
            correct_key = int(item["answer"])
            correct_answer = item["choices"][correct_key]
            if incompatible_answers(correct_answer):
                print(f"Skipping question with incompatible answer: {correct_answer}")
                num_skips += len(item["choices"]) - 1
                continue

            candidates = item["choices"]
            candidates.remove(correct_answer)

            # sample negative answer from remaining choices
            negative_answer = random.choice(candidates)
            if incompatible_answers(negative_answer):
                print(f"Skipping question with incompatible answer: {negative_answer}")
                num_skips += 1
                continue

            # get prompts
            prompt = """Here are a set of example questions and answers\n"""
            for _ in range(n_shot):
                train_item = train_dataset[prompt_idx % len(train_dataset)]
                prompt += f"{raw_to_question(train_item)}\n"
                train_answer = train_item["choices"][train_item["answer"]]
                endings = ""
                negative_train_answer = [ans for ans in train_item["choices"] if ans != train_answer][0]
                new_key = random.randint(0, 1)
                if new_key:
                    endings = f"A) {train_answer} \nB) {negative_train_answer}"
                    answer = "A"
                else:
                    endings = f"A) {negative_train_answer} \nB) {train_answer}"
                    answer = "B"
                prompt += f"{endings} \n"
                prompt += f"Answer: {answer}\n\n"
                prompt_idx += 1

            # print("---- 0-shot question (+ labeled answers) ----")
            # print(question)
            # print("Correct answer: ", correct_answer)
            # print("False answer: ", negative_answer)
            # print("---- 5-shot example ----")
            # print(prompt)
            # print("---- COT example ----")
            # print(cot_prompt)
            # print("----")

            # raise ValueError
            csv_writer.writerow(
                [
                    source_id,
                    question,
                    prompt,
                    correct_answer,
                    negative_answer,
                    False,
                    "",
                    "",
                    cot_prompt,
                    config,
                ]
            )
    print(f"Skipped {num_skips} questions")
