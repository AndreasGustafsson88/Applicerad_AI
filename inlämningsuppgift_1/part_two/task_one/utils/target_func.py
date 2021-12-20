import numpy as np
from math import e


def carl_func2(x: float, y: float) -> float:
    """Reverse function to locate max value"""
    return -1*e**(-0.05*(x**2 + y**2)) * (np.arctan(x) - np.arctan(y) + e**(-1*(x**2 + y**2))*(np.cos(x)**2 * np.sin(y)**2))
