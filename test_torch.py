import torch

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
