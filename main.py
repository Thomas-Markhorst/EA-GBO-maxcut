import numpy as np
from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction
from plot import InitPlots

all_instances = np.array([
    ["A", "06i01", "maxcut-instances/setA/n0000006i01.txt"],
    ["A", "12i01", "maxcut-instances/setA/n0000012i01.txt"],
    ["A", "25i01", "maxcut-instances/setA/n0000025i01.txt"],
    ["A", "50i01", "maxcut-instances/setA/n0000050i01.txt"],
    ["A", "100i01", "maxcut-instances/setA/n0000100i01.txt"],
    ["B", "09i01", "maxcut-instances/setB/n0000009i01.txt"],
    ["B", "16i01", "maxcut-instances/setB/n0000016i01.txt"],
    ["B", "25i01", "maxcut-instances/setB/n0000025i01.txt"],
    ["B", "49i01", "maxcut-instances/setB/n0000049i01.txt"],
    ["B", "100i01", "maxcut-instances/setB/n0000100i01.txt"],
    ["C", "06i01", "maxcut-instances/setC/n0000006i01.txt"],
    ["C", "12i01", "maxcut-instances/setC/n0000012i01.txt"],
    ["C", "25i01", "maxcut-instances/setC/n0000025i01.txt"],
    ["C", "50i01", "maxcut-instances/setC/n0000050i01.txt"],
    ["C", "100i01", "maxcut-instances/setC/n0000100i01.txt"],
    ["D", "10i01", "maxcut-instances/setD/n0000010i01.txt"],
    ["D", "20i01", "maxcut-instances/setD/n0000020i01.txt"],
    ["D", "40i01", "maxcut-instances/setD/n0000040i01.txt"],
    ["D", "80i01", "maxcut-instances/setD/n0000080i01.txt"],
    ["D", "160i01", "maxcut-instances/setD/n0000160i01.txt"],
    ["E", "10i01", "maxcut-instances/setE/n0000010i01.txt"],
    ["E", "20i01", "maxcut-instances/setE/n0000020i01.txt"],
    ["E", "40i01", "maxcut-instances/setE/n0000040i01.txt"],
    ["E", "80i01", "maxcut-instances/setE/n0000080i01.txt"],
    ["E", "160i01", "maxcut-instances/setE/n0000160i01.txt"]
])

instances = np.array([
    # ["A", 6, "maxcut-instances/setA/n0000006i01.txt"],
    # ["B", 9, "maxcut-instances/setB/n0000009i01.txt"],
    # ["C", 6, "maxcut-instances/setC/n0000006i01.txt"],
    ["D", 10, "maxcut-instances/setD/n0000010i01.txt"],
    ["E", 10, "maxcut-instances/setE/n0000010i01.txt"]
])

# instances = np.array([
#     ["D", 10, "maxcut-instances/setD/n0000010i01.txt"],
#     ["D", 20, "maxcut-instances/setD/n0000020i01.txt"],
#     ["D", 40, "maxcut-instances/setD/n0000040i01.txt"],
#     ["D", 80, "maxcut-instances/setD/n0000080i01.txt"],
#     ["D", 160, "maxcut-instances/setD/n0000160i01.txt"],
#     ["E", 10, "maxcut-instances/setE/n0000010i01.txt"],
#     ["E", 20, "maxcut-instances/setE/n0000020i01.txt"],
#     ["E", 40, "maxcut-instances/setE/n0000040i01.txt"],
#     ["E", 80, "maxcut-instances/setE/n0000080i01.txt"],
#     ["E", 160, "maxcut-instances/setE/n0000160i01.txt"]
# ])


if __name__ == "__main__":
    # settings:
    n = int(len(instances))
    num_populations = 10
    crossovers = ["CustomCrossover", "UniformCrossover", "OnePointCrossover"]
    bar_plot = True
    pop_plot = True
    log_file = True

    draw_bar = InitPlots().initbar(bar_plot, crossovers, n)

    if log_file:
        for cx in crossovers:
            with open("output-{}.txt".format(cx), "w") as f:
                f.write("\n Layout: instance, population size, success rate, percentiles1, percentiles2, percentiles3, "
                        "best fitness \n")

    # main loop:
    for s in range(n):
        inst = instances[s, 2]

        opt_succesrate = np.zeros(len(crossovers))
        opt_medianeval = np.zeros(len(crossovers))
        opt_fitness = np.zeros(len(crossovers))

        population = np.round(10*np.linspace(1, 3, num_populations)*int(0.5*int(instances[s, 1]))-1*np.ones(num_populations))
        print(population)
        for w in range(num_populations):
            if 100000 % population[w] != 0 or population[w] % 2 != 0:
                if w >= 1:
                    if population[w-1] >= population[w]:
                        population[w] = population[w-1] + 1
                while 100000 % population[w] != 0 or population[w] % 2 != 0:
                    population[w] += 1
        print(population)

        draw_plots = InitPlots().initplot(pop_plot, crossovers, num_populations)

        for p in range(len(population)):
            population_size = int(population[p])

            for cx in crossovers:
                num_evaluations_list = []
                num_runs = 30
                num_success = 0
                best_fitness = 0

                for i in range(num_runs):
                    fitness = FitnessFunction.MaxCut(inst)
                    genetic_algorithm = GeneticAlgorithm(fitness, population_size, variation=cx, evaluation_budget=100000,
                                                         verbose=False)
                    best_fitness, num_evaluations = genetic_algorithm.run()

                    if best_fitness == fitness.value_to_reach:
                        num_success += 1

                    num_evaluations_list.append(num_evaluations)

                print("Instance {}, Crossover method {}".format(instances[s, 0] + instances[s, 1], cx))
                print("{}/{} runs successful".format(num_success, num_runs))
                print("{} evaluations (median)".format(np.median(num_evaluations_list)))
                print("{} best fitness".format(best_fitness))

                # set optimals, should it be optimal over all population sizes?
                if np.median(num_evaluations_list) > opt_medianeval[crossovers.index(cx)]:
                    opt_medianeval[crossovers.index(cx)] = np.median(num_evaluations_list)
                if num_success / num_runs > opt_succesrate[crossovers.index(cx)]:
                    opt_succesrate[crossovers.index(cx)] = num_success / num_runs
                if best_fitness > opt_fitness[crossovers.index(cx)]:
                    opt_fitness[crossovers.index(cx)] = best_fitness

                if bar_plot and p == num_populations-1:
                    # inst_num, instance, crossovers, cx_num, median, success, fitness
                    draw_bar.make_bar_plots(s, instances[s, 0] + instances[s, 1],
                                            crossovers.index(cx), opt_medianeval[crossovers.index(cx)],
                                            opt_succesrate[crossovers.index(cx)], opt_fitness[crossovers.index(cx)])

                if pop_plot:
                    # instance, cx_num, median, success, fitness, population):
                    draw_plots.make_plots(instances[s, 0] + instances[s, 1], crossovers.index(cx),
                                    np.median(num_evaluations_list), num_success/num_runs, best_fitness, population_size)

                if log_file:
                    with open("output-{}.txt".format(cx), "w") as f:
                        percentiles = np.percentile(num_evaluations_list, [10, 50, 90])
                        f.write("{} {} {} {} {} {}\n".format(instances[s, 0] + instances[s, 1], population_size,
                                                             num_success / num_runs, percentiles[0], percentiles[1],
                                                             percentiles[2], best_fitness))
