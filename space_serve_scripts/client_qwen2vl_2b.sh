python ../benchmarks/benchmark_serving_6_6.py \
        --backend openai-chat \
        --model Qwen/Qwen2-VL-2B-Instruct \
        --dataset-name hf --hf-subset vision \
        --dataset-path MMMU/MMMU_Pro \
        --request-rate 10 \
        --num-prompts 200 \
        --hf-split test \
        --endpoint /chat/completions \
      --base-url http://127.0.0.1:8000/v1
