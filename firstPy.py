import torch
import time

# Check for Apple Silicon GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Create two large 5000x5000 matrices
size = 5000
x = torch.randn(size, size)
y = torch.randn(size, size)

# --- CPU Benchmark ---
start_cpu = time.time()
result_cpu = torch.matmul(x, y)
end_cpu = time.time()
print(f"CPU Time: {end_cpu - start_cpu:.4f} seconds")

# --- GPU (MPS) Benchmark ---
x_gpu = x.to(device)
y_gpu = y.to(device)

# Warm-up run for the GPU
#_ = torch.matmul(x_gpu, y_gpu)

start_gpu = time.time()
result_gpu = torch.matmul(x_gpu, y_gpu)
# Wait for GPU to finish
torch.mps.synchronize() 
end_gpu = time.time()

print(f"GPU Time: {end_gpu - start_gpu:.4f} seconds")