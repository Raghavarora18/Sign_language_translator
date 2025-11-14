# src/utils_gpu_test.py

import sys
import torch
import time

def print_header(title):
    print("\n" + "="*10 + f" {title} " + "="*10)

def gpu_info():
    print_header("PyTorch + CUDA Info")
    print("Python:", sys.version.splitlines()[0])
    print("PyTorch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    try:
        cudnn_version = torch.backends.cudnn.version()
    except Exception:
        cudnn_version = None
    print("cuDNN version (reported by torch):", cudnn_version)

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print("CUDA device count:", n)
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024**3)
            print(f"Device {i}: {name}")
            print(f"  Total memory (GB): {total_gb:.2f}")
            print(f"  MultiProcessorCount: {props.multi_processor_count}")
            print(f"  Major/Minor: {props.major}.{props.minor}")
            print(f"  Device capability: {props.major}.{props.minor}")
    print("-"*40)

def small_matrix_test():
    print_header("Small GPU compute test (matrix multiply)")
    if not torch.cuda.is_available():
        print("Skipped: CUDA not available.")
        return
    try:
        dev = torch.device("cuda:0")
        # modest size to avoid OOM on smaller GPUs
        a = torch.randn(2048, 512, device=dev)
        b = torch.randn(512, 1024, device=dev)
        t0 = time.time()
        c = a @ b
        torch.cuda.synchronize()
        t1 = time.time()
        print("Result shape:", c.shape)
        print(f"Matrix multiply time: {(t1-t0):.4f} sec")
        # free memory
        del a, b, c
        torch.cuda.empty_cache()
    except Exception as e:
        print("Matrix test failed:", repr(e))

def small_model_forward():
    print_header("Small model forward pass test (Conv2d)")
    import torch.nn as nn
    if not torch.cuda.is_available():
        print("Skipped: CUDA not available.")
        return
    try:
        dev = torch.device("cuda:0")
        model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        ).to(dev)
        x = torch.randn(4, 3, 128, 128, device=dev)
        t0 = time.time()
        out = model(x)
        torch.cuda.synchronize()
        t1 = time.time()
        print("Model output shape:", out.shape)
        print(f"Forward time: {(t1-t0):.4f} sec")
        del model, x, out
        torch.cuda.empty_cache()
    except Exception as e:
        print("Model forward test failed:", repr(e))

def quick_cpu_test():
    print_header("Quick CPU sanity test")
    a = torch.randn(10, 10)
    b = torch.randn(10, 10)
    print("CPU matmul ok, shape:", (a @ b).shape)

if __name__ == "__main__":
    gpu_info()
    small_matrix_test()
    small_model_forward()
    quick_cpu_test()
    print("\nFinished. If CUDA is available and device name printed above, PyTorch can use your GPU.")
