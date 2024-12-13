#!/bin/bash
set -e

# Assign command-line arguments with START defaulting to 1 if not provided
RUNS=$1
CONFIG=$2
START=${3:-1} # Default value of 1 if the third argument is not provided

# Initialize an array to store experiment paths
declare -a EXP_PATHS=()

for run in $(seq "$START" "$RUNS"); do
    # while IFS= read -r line; do
    #     echo "$line"         # Print each line for immediate feedback
    #     EXP_PATHS+=("$line") # Append to array
    # done < <(
    python -u train_multimodal.py --config "$CONFIG" --run "$run"
    # )

    # Exit if not successful
    if [ $? -ne 0 ]; then
        echo "Experiment failed.
        Exiting..."
        exit 1
    fi

done
