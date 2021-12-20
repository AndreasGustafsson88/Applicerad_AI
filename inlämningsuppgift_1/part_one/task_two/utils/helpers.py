import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def read_data(path: str) -> pd.DataFrame:
    """Opens file based on path"""
    with open(path, 'r') as f:
        return pd.read_csv(f, index_col=0)


def get_matrix(data: pd.DataFrame) -> np.array:
    """Creates a distance matrix based on distances between cities"""
    return pd.pivot_table(data, values='Distance', index='Start', columns='Target').to_numpy()


def visualize(data: list, n: int = 50) -> None:
    """Plots histogram with n number of bars, sets best result as title"""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(data, bins=[i for i in range(min(data), max(data), (max(data) - min(data)) // n)])
    ax.set_title(f'Best route {min(data)}')
    plt.show()


def visualize_ant_colony(history: list[int]) -> None:
    """Plots history of best result from ant_colony opt."""
    fig, ax = plt.subplots()
    pd.DataFrame(history).cummin().plot(ax=ax)
    plt.show()
