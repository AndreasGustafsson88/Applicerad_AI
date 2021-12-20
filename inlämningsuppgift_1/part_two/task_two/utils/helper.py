import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform, pdist
import numpy as np
from pathlib import Path


def haversine_np(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees)
    based on haversine alg.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def tidy_up(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts minutes to degrees and sets correct appendix +/- depending on North, South, East or West.
    Saves and returns dataframe.
    """
    df.columns = ['city', 'lat', 'lat_min', 'long', 'long_min', 'time']
    df.loc[df['lat_min'].str.contains('S'), 'lat'] *= -1
    df.loc[df['long_min'].str.contains('W'), 'long'] *= -1
    df['lat_min'] = df['lat_min'].map(lambda x: int(x.split()[0]) / 60)
    df['long_min'] = df['long_min'].map(lambda x: int(x.split()[0]) / 60)
    df['lat'] = df['lat'] + df['lat_min']
    df['long'] = df['long'] + df['long_min']
    df.drop(['time', 'lat_min', 'long_min'], axis=1, inplace=True)
    df.to_csv('C:\Kod\Skolkod\\applicerad_AI\inlämningsuppgift_1\data\\dist.csv')
    return df


def read_data() -> pd.DataFrame:
    """
    Reads file if it exists, else fetches and saves from address.
    """
    p = Path('C:\Kod\Skolkod\\applicerad_AI\inlämningsuppgift_1\data\dist.csv')

    if p.exists():
        return pd.read_csv(p, index_col=0)
    else:
        return tidy_up(pd.read_html('https://www.infoplease.com/world/geography/major-cities-latitude-longitude-and-corresponding-time-zones')[0])


def get_matrix(df: pd.DataFrame) -> np.array:
    """
    Transform dataframe to distance matrix with help of squareform, pdist and haversine func.
    Transform and returns numpy array with ints.
    """
    return pd.DataFrame(squareform(pdist(df.iloc[:, 1:], lambda x, y: haversine_np(*x, *y)))).to_numpy(dtype=int)


def visualize(data: list, n: int = 50) -> None:
    """Plots histogram with n number of bars, sets best result as title"""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(data, bins=[i for i in range(min(data), max(data), (max(data) - min(data)) // n)])
    ax.set_title(f'Best route {min(data)}')
    plt.show()


def visualize_ant_colony(history: list) -> None:
    """Plots history of best result from ant_colony opt."""
    fig, ax = plt.subplots()
    pd.DataFrame(history).cummin().plot(ax=ax)
    plt.show()
