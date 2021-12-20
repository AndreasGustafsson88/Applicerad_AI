from inlämningsuppgift_1.part_one.task_one.utils.target_func import carl_func
from inlämningsuppgift_1.part_one.task_one.utils.visualize_func import plot_func
from sko.PSO import PSO
from sko.GA import GA


if __name__ == "__main__":
    ##############
    #  SETTINGS  #
    ##############
    GENERATIONS = 800
    POPULATION = 40
    INERTIA_FACTOR = 0.8
    COGNITIVE_C = 0.5
    SWARM_C = 0.5
    domain = [(-5, 5)] * 2
    ##############

    # Initialize Particle Swarm Opt
    pso = PSO(
        func=carl_func,
        n_dim=len(domain),
        pop=POPULATION,
        max_iter=GENERATIONS,
        lb=[domain[0][0], domain[1][0]],
        ub=[domain[0][1], domain[1][1]],
        w=INERTIA_FACTOR,
        c1=COGNITIVE_C,
        c2=SWARM_C
    )
    pso.run()

    # Initialize Genetic Opt
    ga = GA(
        func=carl_func,
        n_dim=len(domain),
        size_pop=POPULATION,
        max_iter=GENERATIONS,
        lb=[domain[0][0], domain[1][0]],
        ub=[domain[0][1], domain[1][1]],
        precision=1e-7)

    best_x, best_y = ga.run()

    print(' Genetic Optimization '.center(50, '='))
    print('Co-ordinates:', f'{best_x[0]:.05f}, {best_x[1]:.05f} \nMax Value:', f'{-1 * best_y[0]:.05f}')
    print(' Particle Swarm Optimization '.center(50, '='))
    print(f'Co-ordinates: {pso.gbest_x[0]:.05f}, {pso.gbest_x[1]:.05f}')
    print(f'Max Value: {pso.gbest_y[0] * -1:.05f}')

    # VISUALIZE #
    plot_func(domain, carl_func)
    #############
