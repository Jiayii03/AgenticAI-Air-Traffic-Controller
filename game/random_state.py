import random
import numpy as np
import sys

# Global seed that will be used throughout the game
GLOBAL_SEED = 0

def set_global_seed(seed):
    """
    Set the global seed and initialize random with it.
    If seed is 0, generate a random seed.
    Otherwise, use the provided seed.
    
    Returns:
        int: The actual seed used (useful when a random seed was generated)
    """
    global GLOBAL_SEED
    
    # Special case: generate a random seed if seed is 0
    if seed == 0:
        # Generate a seed between 1 and 1,000,000
        seed = random.randint(1, 1000000)
    
    # Set the global seed
    GLOBAL_SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Set torch seed if using PyTorch
    if 'torch' in sys.modules:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    # Make PyTorch operations deterministic
    if 'torch' in sys.modules:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Global random seed set to: {seed}")
    
    return seed

def get_global_seed():
    """Get the current global seed"""
    return GLOBAL_SEED