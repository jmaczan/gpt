import torch
import time

# Check device availability
mps_available = torch.backends.mps.is_available()
print(f"MPS available: {mps_available}")
device_mps = torch.device("mps") if mps_available else torch.device("cpu")
device_cpu = torch.device("cpu")

# Create a random tensor
x = torch.randn(1000, 1000)

# CPU benchmark
x_cpu = x.to(device_cpu)
start = time.time()
for _ in range(100):
    y = x_cpu @ x_cpu
torch.cuda.synchronize() if torch.cuda.is_available() else None
print(f"CPU time: {time.time() - start:.4f} sec")

# MPS benchmark
if mps_available:
    x_mps = x.to(device_mps)
    start = time.time()
    for _ in range(100):
        y = x_mps @ x_mps
    torch.mps.synchronize()
    print(f"MPS time: {time.time() - start:.4f} sec")
else:
    print("MPS is not available")
