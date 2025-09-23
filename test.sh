#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=6,7

LOG_DIR="logs"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/test_$(date +'%Y%m%d_%H%M%S').log"

nohup python evaluate_all_checkpoints.py \
    --config "configs/multigpu.yaml" \
    --checkpoint_dir "data/outputs/2025.09.18/17.36.30_train_diffusion_unet_hybrid_multi_camera_stack_cube_ur10e_image" \
    > "${LOG_FILE}" 2>&1 &

TEST_PID=$!

# Write PID information to the log file
echo "================== Testing Process Information ==================" >> "${LOG_FILE}"
echo "Testing started at: $(date)" >> "${LOG_FILE}"
echo "Testing PID: $TEST_PID" >> "${LOG_FILE}"
echo "Log file: $LOG_FILE" >> "${LOG_FILE}"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES" >> "${LOG_FILE}"
echo "Checkpoint directory: data/outputs/2025.09.18/17.36.30_train_diffusion_unet_hybrid_multi_camera_stack_cube_ur10e_image" >> "${LOG_FILE}"
echo "=================================================================" >> "${LOG_FILE}"
echo "" >> "${LOG_FILE}"

# Also display on console
echo "Testing started with PID: $TEST_PID"
echo "Log file: $LOG_FILE"
echo "To monitor the process: ps -p $TEST_PID"
echo "To kill the process: kill $TEST_PID"
echo "To view logs: tail -f $LOG_FILE"