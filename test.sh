#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="1"

CONFIG_PATH="configs/multiobj.yaml"
CHECKPOINT_DIR="data/outputs/2025.12.23/22.32.31_train_diffusion_unet_hybrid_multi_camera_stack_cube_ur10e_image"

LOG_DIR="logs"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/test_$(date +'%Y%m%d_%H%M%S').log"

nohup python evaluate_all_checkpoints.py \
    --config "${CONFIG_PATH}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    > "${LOG_FILE}" 2>&1 &

TEST_PID=$!

# Write PID information to the log file
echo "================== Testing Process Information ==================" &>> "${LOG_FILE}"
echo "Testing started at: $(date)" &>> "${LOG_FILE}"
echo "Testing PID: $TEST_PID" &>> "${LOG_FILE}"
echo "Log file: $LOG_FILE" &>> "${LOG_FILE}"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES" &>> "${LOG_FILE}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}" &>> "${LOG_FILE}"
echo "=================================================================" &>> "${LOG_FILE}"
echo "" &>> "${LOG_FILE}"

# Also display on console
echo "Testing started with PID: $TEST_PID"
echo "Log file: $LOG_FILE"
echo "To monitor the process: ps -p $TEST_PID"
echo "To kill the process: kill $TEST_PID"
echo "To view logs: tail -f $LOG_FILE"