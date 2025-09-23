#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=6,7

LOG_DIR="logs"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/train_$(date +'%Y%m%d_%H%M%S').log"

nohup python train.py \
  --config-dir=configs \
  --config-name=multigpu.yaml \
  training.seed=42 \
  hydra.run.dir="data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}" \
  > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!

# Write PID information to the log file
echo "================== Training Process Information ==================" >> "${LOG_FILE}"
echo "Training started at: $(date)" >> "${LOG_FILE}"
echo "Training PID: $TRAIN_PID" >> "${LOG_FILE}"
echo "Log file: $LOG_FILE" >> "${LOG_FILE}"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES" >> "${LOG_FILE}"
echo "=================================================================" >> "${LOG_FILE}"
echo "" >> "${LOG_FILE}"

# Also display on console
echo "Training started with PID: $TRAIN_PID"
echo "Log file: $LOG_FILE"
echo "To monitor the process: ps -p $TRAIN_PID"
echo "To kill the process: kill $TRAIN_PID"
echo "To view logs: tail -f $LOG_FILE"
