#!/usr/bin/env bash
# Boot semif: install SemIf + llama-cpp-python (CUDA 12 wheel), patch SemIf's hard-coded CPU-only
# flag, fetch the Q4 GGUF, pick a GPU, start the service.
# Tested on the vllm/vllm-openai:v0.29.0 image (torch, transformers and CUDA already present; it
# ships CUDA 13, hence the cu12 runtime libs below). Any CUDA image with Python >= 3.10 and
# transformers should work. Mount /models as a volume so restarts skip the downloads.
set -e
export HF_HOME=/models/hf-cache PIP_CACHE_DIR=/models/pip-cache HF_HUB_DISABLE_PROGRESS_BARS=1
pip install -q --no-deps https://github.com/TheoLeeCJ/SemIf/archive/${SEMIF_REF:-master}.tar.gz 2>&1 | grep -v WARNING | tail -1
pip install -q nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 2>&1 | grep -v WARNING | tail -1
SP=$(python3 -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$(dirname $(find $SP/nvidia -name "libcudart.so.12*" | head -1)):$(dirname $(find $SP/nvidia -name "libcublas.so.12*" | head -1)):$(dirname $(find $SP/nvidia -name "libcublasLt.so.12*" | head -1)):$LD_LIBRARY_PATH
pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 2>&1 | grep -v WARNING | tail -1
# ⚠️ SemIf's llama.cpp backend ships with n_gpu_layers = 0: ~1.8 s per decision on CPU vs ~61 ms on GPU.
B=$(python3 -c "import semif_phase1.llamacpp_backend as m; print(m.__file__)"); sed -i 's/n_gpu_layers = 0/n_gpu_layers = 99/; s/"n_gpu_layers": 0/"n_gpu_layers": 99/' "$B"
python3 -c "import llama_cpp; assert llama_cpp.llama_supports_gpu_offload(), 'no GPU offload'"
export GGUF=$(python3 -c "
from huggingface_hub import hf_hub_download; print(hf_hub_download('bartowski/Qwen_Qwen3.5-4B-GGUF','Qwen_Qwen3.5-4B-Q4_K_M.gguf'))")
# Pick a card by BEST FIT: the tightest card that still has NEED_MIB free. Sharing a card with your
# LLM is fine (it needs ~2.7 GB); best fit keeps the biggest free gap whole for something larger.
NEED_MIB="${NEED_MIB:-3500}"
read -r GPU_UUID GPU_FREE <<<"$(nvidia-smi --query-gpu=uuid,memory.used,memory.total --format=csv,noheader,nounits | awk -F", " -v need="$NEED_MIB" '{f=$3-$2; if(f>=need && (b=="" || f<b)){b=f; u=$1}} END{print u, b}')"
[ -n "$GPU_UUID" ] || { echo "  no card has ${NEED_MIB} MiB free"; nvidia-smi --query-gpu=uuid,memory.used,memory.total --format=csv,noheader; exit 3; }
export CUDA_VISIBLE_DEVICES="$GPU_UUID"; echo "  using $GPU_UUID (${GPU_FREE} MiB free)"
exec python3 "$(dirname "$0")/semif_service.py"
