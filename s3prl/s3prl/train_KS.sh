#!/bin/bash

# Ensure that the user has provided a GPU device ID
if [ -z "$1" ]; then
    echo "Usage: $0 <GPU_DEVICE>"
    exit 1
fi

# Set the model name (e.g., wavlm+mert_hybrid or wavlm+mert_lconv)
MODEL_NAME="wavlm+mert_0.3_hybrid"

# GPU device ID passed as an argument
GPU_DEVICE=$1

# Base command to train a model
BASE_CMD="python run_downstream.py -m train -u multi_distiller_local -d speech_commands -s paper -k result/pretrain/${MODEL_NAME}/states-epoch-12.ckpt -g result/pretrain/${MODEL_NAME}/config_model.yaml"

# Create an array to store test accuracy for each dataset
declare -a TEST_ACCURACIES

# Loop through 5 datasets with different directories for speech_commands_root
for i in {1..5}
do
    # Set the root directory for each dataset
    DATASET_DIR="/home/johnwei743251/mdd/dataset/speech_commands/few_shot_${i}"

    # Set a unique experiment name for each run
    EXP_NAME="${MODEL_NAME}_fewshot_20_KS_${i}"

    # Set the CUDA_VISIBLE_DEVICES environment variable for this run
    export CUDA_VISIBLE_DEVICES=$GPU_DEVICE

    # Construct the command with the specific dataset directory and experiment name for this iteration
    CMD="${BASE_CMD} -n ${EXP_NAME} -o config.downstream_expert.datarc.speech_commands_root=${DATASET_DIR}"

    # Print the command for debugging purposes
    echo "Running command for dataset ${i} with model ${MODEL_NAME} on GPU ${GPU_DEVICE}:"
    echo $CMD
    #echo "Using dataset: /home/johnwei743251/mdd/dataset/speech_commands/few_shot_${i}"

    # Execute the training command
    $CMD

    # Now, evaluate the model after training and capture the test accuracy
    EVAL_CMD="python3 run_downstream.py -m evaluate -e result/downstream/${EXP_NAME}/dev-best.ckpt"

    # Capture the output of the evaluation (test accuracy)
    TEST_ACC=$( $EVAL_CMD | grep "test acc" | awk '{print $3}' )

    # Store the test accuracy in the array
    TEST_ACCURACIES+=($TEST_ACC)

    # Print the test accuracy for the current run
    echo "Test accuracy for model ${EXP_NAME}: $TEST_ACC"
done

# After all runs, calculate the average test accuracy
total=0
count=${#TEST_ACCURACIES[@]}
for acc in "${TEST_ACCURACIES[@]}"
do
    total=$(echo "$total + $acc" | bc)
done

average=$(echo "$total / $count" | bc -l)

# Print the test accuracies for all 5 models
echo "Test Accuracies for all 5 models:"
for i in {0..4}
do
    echo "Model $((i+1)): ${TEST_ACCURACIES[$i]}"
done

# Print the average test accuracy
echo "Average test accuracy: $average"

