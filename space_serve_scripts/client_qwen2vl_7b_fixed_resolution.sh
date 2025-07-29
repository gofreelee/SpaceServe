#!/bin/bash

# Client script for Qwen2-VL-7B with fixed image resolution
# This script demonstrates sending images with a fixed resolution to the server

python ../benchmarks/benchmark_serving_6_6.py \
        --backend openai-chat \
        --model Qwen/Qwen2-VL-7B-Instruct \
        --dataset-name hf --hf-subset vision \
        --dataset-path MMMU/MMMU_Pro \
		--request-rate 10 \
        --num-prompts 200 \
        --hf-split test \
        --endpoint /chat/completions \
        --base-url http://127.0.0.1:7777/v1 \
        --fixed-image-resolution 224,224
