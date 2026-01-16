import torch 
import random
import numpy as np

def set_seeds(seed):
    """Set seeds for reproducibility across all random number generators.
    
    Args:
        seed (int): The seed value to use for all random number generators
    """
    # Python's built-in random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Uncomment for additional for full reproducibility, but slower code
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False