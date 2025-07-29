export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps  # 设置 MPS 管道目录

export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log   # 设置 MPS 日志目录

mkdir -p $CUDA_MPS_PIPE_DIRECTORY
mkdir -p $CUDA_MPS_LOG_DIRECTORY

nvidia-cuda-mps-control -d



