# Evals Template

## Overview

This repository contains the `run.py` script and associated files for conducting evaluations using LLMs from the Anthroppic and OpenAI APIs. It is designed to handle tasks such as generating responses to prompts, caching results, and managing API interactions efficiently.

## Setup

### Prerequisites

- Python 3.11
- Virtual environment tool (e.g., virtualenv)

### Installation

1. **Create and Activate a Virtual Environment:**
    ```bash
    virtualenv --python python3.11 .venv
    source .venv/bin/activate
    ```
2. Install Required Packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Install Pre-Commit Hooks:
    ```bash
    make hooks
    ```
4. Create a SECRETS file
    ```bash
    touch SECRETS
    echo OPENAI_API_KEY=<INSERT_HERE> >> SECRETS
    echo ANTHROPIC_API_KEY=<INSERT_HERE> >> SECRETS
    echo ACEDEMICNYUPEREZ_ORG=org-<INSERT_HERE> >> SECRETS
    echo FARAI_ORG=org-<INSERT_HERE> >> SECRETS
    ```

## Usage
### Running the Script

- **Basic Run on the MMLU Dataset:**
    You must specify an experiment directory to store results. All the logs and hydra config for that experiment will be automatically saved there.
    Check out teh default config in `evals/conf/config.yaml` for more options.
    ```bash
    python3 -m evals.run ++exp_dir=exp/test_run1
    ```
- **Advanced Usage with Overrides:**
    To test a different model, limit to 5 samples, and print output:
    ```bash
    python3 -m evals.run ++exp_dir=exp/test_run2 ++limit=5 ++print_prompt_and_response=true ++language_model.model=gpt-3.5-turbo-instruct ++reset=true
    ```
- **Creating and using a new prompt:**
    If you want to create a new prompt you can add it to the prompt folder. E.g. creating a prompt woth chain of thought `evals/conf/prompt/cot.yaml` and then use it instead of zero-shot like this:
    ```bash
    python3 -m evals.run ++exp_dir=exp/test_run3 prompt=cot
    ```
    A prompt files contains a messages field that has the prompt in the standard openAI messages format. You can use string templating e.g. `$question` within the prompt which can be filled in via the code.
- **Creating and using a LLM config:**
    You can create new specific LLM param configs so you don't have to override lots of parameters. E.g. creating a config for gpt-4 with temperature 0.8 in `evals/conf/language_model/gpt-4-temp-0.8.yaml`:
    ```bash
    python3 -m evals.run ++exp_dir=exp/test_run5 ++language_model=gpt-4-temp-0.8
    ```
- **Control your API usage:**
    Control number of threads with `anthropic_num_threads` and `openai_fraction_rate_limit` which you can set via the command line or in the config file.

### Features

- **Hydra for Configuration Management:**
  Hydra enables easy overriding of configuration variables. Use `++` for overrides. You can reference other variables within variables using `${var}` syntax.

- **Caching Mechanism:**
  Caches prompt calls to avoid redundant API calls. Cache location defaults to `$exp_dir/cache`.

- **Prompt History Logging:**
  For debugging, human-readable `.txt` files are stored in `$exp_dir/prompt_history`, timestamped for easy reference.

- **LLM Inference API Enhancements:**
  - Ability to double the rate limit if you pass a list of models e.g. ["gpt-3.5-turbo", "gpt-3.5-turbo-0613"]
  - Manages rate limits efficiently, bypassing the need for exponential backoff.
  - Allows custom filtering of responses via `is_valid` function.
  - Provides a running total of cost and model timings for performance analysis.
  - Utilise maximum rate limit by setting `max_tokens=None` for OpenAI models.

- **Logging finetuning runs with Weights and Biases:**
  - Logs finetuning runs with Weights and Biases for easy tracking of experiments.

- **Usage Tracking:**
  - Tracks usage of OpenAI and Anthropic APIs so you know how much they are being utilised within your organisation.

## Repository Structure

- `evals/run.py`: Main script for evaluations.
- `evals/apis/inference`: Directory containing modules for LLM inference
- `evals/apis/finetuning`: Directory containing scripts to finetune OpenAI models and log with weights and biases
- `evals/apis/usage`: Directory containing two scripts to get usage information from OpenAI and Anthropic
- `evals/conf`: Directory containing configuration files for Hydra. Check out `prompt` and `language_model` for examples of how to create useful configs.
- `evals/data_models`: Directory containing Pydantic data models
- `evals/load`: Directory containing code to download and process MMLU
- `tests`: Directory containing unit tests

## Contributing

Contributions to this repository are welcome. Please follow the standard procedures for submitting issues and pull requests.

---
