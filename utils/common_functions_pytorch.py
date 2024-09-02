"""Some common functions with a pytorch dependency"""
import os

import psutil
import torch

def print_cpu_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"CPU Memory used: {process.memory_info().rss / 1024**3:.2f} GB")
def print_gpu_memory_usage():
    memory = torch.cuda.memory_allocated()
    print(f"GPU Memory used: {memory / 1024**3:.2f} GB")

def print_memory_usage():
    print_cpu_memory_usage()
    print_gpu_memory_usage()