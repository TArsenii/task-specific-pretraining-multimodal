#!/bin/bash
set -e
RUNS=$1
CONFIG=$2

# Initialize an array to store experiment paths
declare -a EXP_PATHS=()

for run in $(seq 1 "$RUNS"); do
    while IFS= read -r line; do
        echo "$line"         # Print each line for immediate feedback
        EXP_PATHS+=("$line") # Append to array
    done < <(python -u train_multimodal.py --config "$CONFIG" --run "$run")
done
