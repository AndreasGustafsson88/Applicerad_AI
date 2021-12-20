import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot_func(val_range: list[tuple[int, int]], func) -> None:
    """Visualize function, Z = -1 to plot correct"""
    plt.rcParams["figure.figsize"] = (20, 20)

    x = y = np.linspace(*val_range[0], 1000)

    X, Y = np.meshgrid(x, y)
    Z = -1 * func(X, Y)  # INVERTED

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=Axes3D.name)

    surf = ax.plot_surface(Y, X, Z, cmap=cm.coolwarm, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
