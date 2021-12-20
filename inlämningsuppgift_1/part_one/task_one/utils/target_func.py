import numpy as np


def carl_func(x: float, y: float) -> float:
    """Reverse function to find max val"""
    return -1 * ((np.cos(x) + np.sin(y)) ** 2 / (1 + abs(x) + abs(y)))
