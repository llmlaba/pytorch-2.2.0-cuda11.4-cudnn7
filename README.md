# PyTorch 2.2.0 for Kepler - CUDA 11.4 Compute Capability 3.7 (Ubuntu 20.04)

A reproducible container environment for **Tesla K80 (Kepler, sm_37)** that runs **PyTorch 2.2.0** on **CUDA 11.4** with **cuDNN 8** and **Python 3.8**. Built from source with `TORCH_CUDA_ARCH_LIST="3.7"` to support Kepler.

> ⚠️ Note: CUDA 11.4 for PyTorch 2.2 is **outside PyTorch's official CUDA matrix** (they target 11.8/12.1). This image is an enthusiast setup, verified with NVIDIA driver **470.xx** (Kepler supported).

## Why this exists

- Keep old, cheap GPUs (K80) useful with a modern-enough PyTorch.
- Run `transformers`, `diffusers` (e.g. **Stable Diffusion 1.5**), and LLMs (e.g. **Mistral-7B** with 2×K80 sharding + CPU/disk offload).
- Isolate CUDA/cuDNN/Python userspace to avoid host conflicts and make runs reproducible.

## What’s inside

- **Ubuntu 20.04**
- **CUDA 11.4.3**, **cuDNN 8 (devel)**
- **Python 3.8**, `pip`, dev tools (gcc/g++, ninja, cmake, etc.)
- **PyTorch 2.2.0** built from source with `TORCH_CUDA_ARCH_LIST="3.7"`

## Host prerequisites

- NVIDIA driver **470.xx** (Kepler supported; 495+ removed Kepler support)
- Docker
- Preferably **NVIDIA Container Toolkit** — recommended way to expose GPUs (`--gpus all`).
  - If you *don’t* have it, see the **Manual device pass-through** section below.

## Build the image

From the directory with your `Dockerfile`:

```bash
docker build -t kepler .
```

## Run the container

### A With NVIDIA Container Toolkit (recommended)

```bash
docker run --rm -it --gpus all \
  -v $HOME/llm:/llm \
  kepler /bin/bash
```

- `-v $HOME/llm:/llm` mounts your models/scripts into the container at `/llm`.

### B Manual device pass-through (works without Toolkit)

```bash
docker run --rm -it \
  --device=/dev/nvidia0 \
  --device=/dev/nvidia1 \
  --device=/dev/nvidiactl \
  --device=/dev/nvidia-uvm \
  -v $HOME/llm:/llm \
  kepler /bin/bash
```
## First checks inside the container

```bash
# Optional: should print your Kepler GPU
nvidia-smi

python3 - <<'PY'
import torch
print("CUDA avail:", torch.cuda.is_available())
print("CUDA ver:", torch.version.cuda)
print("GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
x = torch.randn(1024,1024, device="cuda"); y = x @ x.t()
print("OK:", y.shape, "cuDNN:", torch.backends.cudnn.is_available(), torch.backends.cudnn.version())
PY
```

## Known limitations

- **Kepler (K80, sm_37)**: no Tensor Cores; skip `flash-attention`, `xformers`, `torch.compile` optimizers.
- Many 8/4‑bit quantizers (**bitsandbytes**, GPTQ/AWQ) target newer GPU archs; expect them not to work on Kepler.
- PyTorch 2.2 + CUDA 11.4 + Kepler is a custom build; officially tested CUDA lines are 11.8/12.1.
- Stay on driver **470.xx** (Kepler removed in future versions).

## Troubleshooting

- **`could not select device driver "" with capabilities: [[gpu]]`**  
  Install NVIDIA Container Toolkit *or* use manual device pass-through as shown above.

- **`nvidia-smi: command not found` inside container**  
  Not required for CUDA to work. If desired, mount the host binary:
  `-v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro`.

- **`no kernel image is available for execution on the device`**  
  Happens if PyTorch was built without `TORCH_CUDA_ARCH_LIST="3.7"` for K80. This image already includes it.

## Licenses

- PyTorch — BSD-style  
- NVIDIA CUDA/cuDNN — NVIDIA proprietary licenses  
