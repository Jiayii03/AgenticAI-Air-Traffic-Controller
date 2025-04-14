import random

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
        print(f"Generated random seed: {seed}")
    
    # Set the global seed
    GLOBAL_SEED = seed
    random.seed(seed)
    print(f"Global random seed set to: {seed}")
    
    return seed

def get_global_seed():
    """Get the current global seed"""
    return GLOBAL_SEED

# Initialize with a default seed (don't set it here to avoid early initialization)
GLOBAL_SEED = 0