import torch

def get_device_name():
    """
    Get the name of the device to use for training.
    Returns 'cuda' if CUDA is available, 'mps' if Apple Silicon is available, or 'cpu' otherwise.
    """
    return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'