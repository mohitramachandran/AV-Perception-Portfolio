
import torch

def check_gpu():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected — running on CPU")

if __name__ == "__main__":
    check_gpu()
