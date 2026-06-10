# Optional FlashAttention2 Setup

FlashAttention2 is optional. It is useful for memory-heavy full-resolution multi-image verification, especially on A100 GPUs.

## Check Environment

```bash
python - <<'PY'
import torch, sys
print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("cxx11 abi:", torch._C._GLIBCXX_USE_CXX11_ABI)
PY
```

## Install

Install a wheel matching your Python, PyTorch, CUDA, and CXX11 ABI. For example, with Python 3.10, PyTorch 2.5, CUDA 12, ABI false:

```bash
pip install flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-deps
```

Verify:

```bash
python - <<'PY'
import flash_attn
print(flash_attn.__version__)
from flash_attn.flash_attn_interface import flash_attn_func
print("flash_attn import ok")
PY
```

## Use in Transformers

Pass:

```python
attn_implementation="flash_attention_2"
```

to `Qwen2_5_VLForConditionalGeneration.from_pretrained(...)`.

## Triton libcuda Fix

If you see `cannot find -lcuda`, add a symlink and export library paths:

```bash
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$CONDA_PREFIX/lib/libcuda.so"

export LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
```
