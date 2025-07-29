python ../benchmarks/benchmark_serving_6_6.py \
        --backend openai-chat \
        --model Qwen/Qwen2-VL-7B-Instruct \
        --dataset-name hf --hf-subset vision \
        --dataset-path MMMU/MMMU_Pro \
        --request-rate 20 \
        --num-prompts 20 \
        --hf-split test \
        --endpoint /chat/completions \
      --base-url http://127.0.0.1:7777/v1
