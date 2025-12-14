#!/bin/bash

##################### Paths #####################

# Set default paths
DEFAULT_DATA_DIR="${PWD}/Bench2Drive-Base"
DEFAULT_LOG_DIR="${PWD}/log"
DEFAULT_CAMERA_LABEL_DIR="${PWD}/camera_labels"
PYTHONPATH="${PWD}"
DEFAULT_EXPERIMENT_DIR="${PWD}/exp"
DEFAULT_VLM_PATH="${PWD}/paligemma-3b-pt-224"

# Prompt the user for input, allowing overrides
read -p "Enter the desired Bench2Drive Dataset directory [default: ${DEFAULT_DATA_DIR}], leave empty to use default: " DATA_DIR
DATA_DIR=${DATA_DIR:-$DEFAULT_DATA_DIR}  # Use user input or default if input is empty

read -p "Enter the desired camera labels directory [default: ${DEFAULT_CAMERA_LABEL_DIR}], leave empty to use default: " CAMERA_LABEL_DIR
CAMERA_LABEL_DIR=${CAMERA_LABEL_DIR:-$DEFAULT_CAMERA_LABEL_DIR}  # Use user input or default if input is empty

read -p "Enter the desired logging directory [default: ${DEFAULT_LOG_DIR}], leave empty to use default: " LOG_DIR
LOG_DIR=${LOG_DIR:-$DEFAULT_LOG_DIR}  # Use user input or default if input is empty

read -p "Enter the desired experiment directory [default: ${DEFAULT_EXPERIMENT_DIR}], leave empty to use default: " EXPERIMENT_DIR
EXPERIMENT_DIR=${EXPERIMENT_DIR:-$DEFAULT_EXPERIMENT_DIR}  # Use user input or default if input is empty

read -p "Enter the desired pretrained vlm directory [default: ${DEFAULT_VLM_PATH}], leave empty to use default: " VLM_PATH
VLM_PATH=${VLM_PATH:-$DEFAULT_VLM_PATH}  # Use user input or default if input is empty

# Export to current session
export DATA_DIR="$DATA_DIR"
export LOG_DIR="$LOG_DIR"
export CAMERA_LABEL_DIR="$CAMERA_LABEL_DIR"
export PYTHONPATH="$PYTHONPATH"
export REPO_DIR="${PWD}"
export EXPERIMENT_DIR="$EXPERIMENT_DIR"
export VLM_PATH="$VLM_PATH"

# Confirm the paths with the user
echo "Data directory set to: $DATA_DIR"
echo "Camera label directory set to: $CAMERA_LABEL_DIR"
echo "Log directory set to: $LOG_DIR"
echo "Experiment directory set to: $EXPERIMENT_DIR"
echo "VLM path set to: $VLM_PATH"

# Append environment variables to .bashrc
echo "export DATA_DIR=\"$DATA_DIR\"" >> ~/.bashrc
echo "export CAMERA_LABEL_DIR=\"$CAMERA_LABEL_DIR\"" >> ~/.bashrc
echo "export LOG_DIR=\"$LOG_DIR\"" >> ~/.bashrc
echo "export EXPERIMENT_DIR=\"$EXPERIMENT_DIR\"" >> ~/.bashrc
echo "export VLM_PATH=\"$VLM_PATH\"" >> ~/.bashrc

echo "Environment variables DATA_DIR, CAMERA_LABEL_DIR, LOG_DIR, EXPERIMENT_DIR and VLM_PATH added to .bashrc and applied to the current session."

##################### WandB #####################

# Prompt the user for input, allowing overrides
read -p "Enter your WandB entity (username or team name), leave empty to skip: " ENTITY

# Check if ENTITY is not empty
if [ -n "$ENTITY" ]; then
  # If ENTITY is not empty, set the environment variable
  export WANDB_ENTITY="$ENTITY"

  # Confirm the entity with the user
  echo "WandB entity set to: $WANDB_ENTITY"

  # Append environment variable to .bashrc
  echo "export WANDB_ENTITY=\"$ENTITY\"" >> ~/.bashrc

  echo "Environment variable WANDB_ENTITY added to .bashrc and applied to the current session."
else
  # If ENTITY is empty, skip setting the environment variable
  echo "No WandB entity provided. Please set wandb=null when running scripts to disable wandb logging and avoid error."
fi