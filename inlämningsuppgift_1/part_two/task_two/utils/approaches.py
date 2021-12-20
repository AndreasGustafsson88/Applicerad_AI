import numpy as np


def greedy_approach(domain: list[tuple[int, int]], func, matrix: np.array, start_idx: int) -> int:
    # Doesn't really need to be run 1000 iterations, since the result is static from every starting point.
    # Only need to test the 120 different starting pos to see which one generates the lowest miles travelled

    path = [start_idx]

    for _ in range(len(domain) - 1):
        city = matrix[path[-1]]
        values_idx = sorted([(i, dist) for i, dist in enumerate(city)], key=lambda x: x[1])
        filtered_values = list(filter(lambda x: x if x[0] not in path else None, values_idx))
        path.append(filtered_values[0][0])

    return func(path)
