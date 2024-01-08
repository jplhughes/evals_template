#!/bin/bash
set -eoux pipefail
DATE=$(date '+%y-%m-%d')

model=gpt-4-1106-preview
model_judge=gpt-4-1106-preview
model_judge_long_context=gpt-4-1106-preview
sweep_dir=./exp/data_${DATE}_model_sweep
anthropic_num_threads=50
openai_fraction_rate_limit=0.5
limit=25
organization=ACEDEMICNYUPEREZ_ORG
standard_args="++anthropic_num_threads=$anthropic_num_threads ++limit=$limit ++openai_fraction_rate_limit=$openai_fraction_rate_limit ++organization=$organization"

for model in gpt-3.5-turbo claude-2.1; do
    for temperature in 0.0 0.3 0.6; do
        exp_dir=${sweep_dir}/${model}_temperature_${temperature}
        python3 -m evals.run $standard_args ++language_model.model=$model ++language_model.temperature=$temperature ++exp_dir=$exp_dir
    done
done
