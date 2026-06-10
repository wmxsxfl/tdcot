# Troubleshooting

## CUDA Is Not Available

Check:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

If `nvidia-smi` fails after reboot, the cloud image may have booted into a kernel incompatible with the installed NVIDIA driver.

## Qwen Video Reading Hangs

Some video paths/codecs can cause video loading to hang. The MVBench direct runner includes an image-frame based path to avoid this issue.

For problematic videos:

- decode frames with `decord`
- save sampled frames
- pass images to Qwen2.5-VL instead of passing a video object

## Out of Memory

Try:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Reduce:

- number of frames
- image resolution
- number of images per verification call

For full-resolution multi-image verification, FlashAttention2 can help.

## FlashAttention2 / Triton libcuda Error

If Triton cannot find `libcuda.so`, locate the 64-bit driver library:

```bash
ldconfig -p | grep libcuda
find /usr /lib -name "libcuda.so*" 2>/dev/null | sort
```

A common fix is:

```bash
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$CONDA_PREFIX/lib/libcuda.so"

export LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
```

## Non-Fatal Warnings

The following warnings are usually non-fatal:

- `do_sample=False` with `temperature` or `top_p`
- `Sliding Window Attention is enabled but not implemented for sdpa`
- `Using a slow image processor`

## Missing Data Paths

Most scripts accept path arguments such as:

```bash
--data-path
--dataset-root
--aux-model
--vl-model
--output
```

Use `--help` to inspect the exact arguments.
