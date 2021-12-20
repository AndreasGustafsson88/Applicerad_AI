from sko.ACA import ACA_TSP
from sko.GA import GA_TSP
from inlämningsuppgift_1.part_two.task_two.utils.approaches import greedy_approach
from inlämningsuppgift_1.part_two.task_two.utils.helper import visualize, visualize_ant_colony, get_matrix, read_data


def fitness_func(routine):
    num_points = len(routine)
    return sum([distance_matrix[routine[i], routine[(i + 1) % num_points]] for i in range(num_points)])


def greedy(iterations, domain, matrix):

    greedy_miles = [greedy_approach(domain, fitness_func, matrix, i) for i in range(min(120, iterations))]
    print(min(greedy_miles))
    # 121958 | 10_000 iterations
    visualize(greedy_miles)


def ant_colony(size_pop, max_iter, matrix):
    aca = ACA_TSP(func=fitness_func, n_dim=len(domain),
                  size_pop=size_pop, max_iter=max_iter,
                  distance_matrix=matrix)

    best_x, best_y = aca.run()
    print(best_y)
    # 126094 | 250 iter , 50 pop
    visualize_ant_colony(aca.y_best_history)


def genetic_salesman(population, max_iter, prob_mut):

    ga_tsp = GA_TSP(func=fitness_func, n_dim=len(domain), size_pop=population, max_iter=max_iter, prob_mut=prob_mut)

    best_points, best_distance = ga_tsp.run()
    print(best_distance)
    # 196662 | 500 iter, 50 pop, 0.1 prob_mut

    visualize_ant_colony(ga_tsp.generation_best_Y)


if __name__ == "__main__":
    ##############
    #  SETTINGS  #
    ##############
    NR_ITERATIONS = 1_000
    POPULATION = 50
    MAX_ITER = 100
    PROB_MUT = 0.1
    ##############

    # Fetch data from website or read from csv if file exists
    data = read_data()

    # Convert data to matrix
    distance_matrix = get_matrix(data)

    # Set domain
    domain = [i for i in range(len(distance_matrix))]

    greedy(NR_ITERATIONS, domain, distance_matrix)
    ant_colony(POPULATION, MAX_ITER, distance_matrix)
    genetic_salesman(POPULATION, MAX_ITER, PROB_MUT)
