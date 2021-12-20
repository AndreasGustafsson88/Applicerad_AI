import numpy as np
from sko.ACA import ACA_TSP
from inlämningsuppgift_1.part_one.task_two.utils.approaches import random_path
from inlämningsuppgift_1.part_one.task_two.utils.helpers import visualize, visualize_ant_colony, get_matrix, read_data


def fitness_func(routine: list[int]) -> int:
    """Iterates over dist_matrix based on val in routine"""
    num_points = len(routine)
    return sum([distance_matrix[routine[i], routine[(i + 1) % num_points]] for i in range(num_points)])


def random(iterations: int, domain: list, verbose: bool = 0) -> list[int]:
    """Calls random_path n number based on iterations, plots results if verbose."""

    random_miles = [random_path(domain, fitness_func) for _ in range(iterations)]
    # 846103 | 10_000 iterations
    if verbose:
        visualize(random_miles)

    return random_miles


def ant_colony(size_pop: int, max_iterations: list[int], matrix: np.array, verbose: bool = 0):
    """
    Calls ACA_TSP with different max_iter based on max_iterations. Stores result from each call and returns it.
    Plots each individual run if verbose.
    """
    y = []
    for iteration in max_iterations:
        aca = ACA_TSP(func=fitness_func, n_dim=len(domain),
                      size_pop=size_pop, max_iter=iteration,
                      distance_matrix=matrix)

        best_x, best_y = aca.run()
        y.append(best_y)
        # 126094 | 250 iter , 50 pop
        if verbose:
            visualize_ant_colony(aca.y_best_history)

    return y


if __name__ == "__main__":
    ##############
    #  SETTINGS  #
    ##############
    NR_ITERATIONS = 1_000
    POPULATION = 50
    MAX_ITER = [50, 100, 150]
    DATA_PATH = 'C:\Kod\Skolkod\\applicerad_AI\inlämningsuppgift_1\data\distances.csv'
    ##############

    # Read csv
    data = read_data(DATA_PATH)

    # Convert data to matrix
    distance_matrix = get_matrix(data)

    # Set domain
    domain = [i for i in range(len(distance_matrix))]

    random_y = random(NR_ITERATIONS, domain)
    ant_y = ant_colony(POPULATION, MAX_ITER, distance_matrix)

    print(' Random Salesman '.center(50, '='))
    print(f'Least travelled miles: {min(random_y)}')

    print(' Ant Travel '.center(50, '='))
    print(f'Least travelled miles: {min(ant_y)}')
