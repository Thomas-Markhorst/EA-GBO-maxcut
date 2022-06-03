import numpy as np
import matplotlib.pyplot as plt
from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction

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

easy_instances = np.array([
    ["A", "06i01", "maxcut-instances/setA/n0000006i01.txt"],
    ["B", "09i01", "maxcut-instances/setB/n0000009i01.txt"],
    ["C", "06i01", "maxcut-instances/setC/n0000006i01.txt"],
    ["D", "10i01", "maxcut-instances/setD/n0000010i01.txt"],
    ["E", "10i01", "maxcut-instances/setE/n0000010i01.txt"]
])

instances = np.array([
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


class PlotResults:

    def __init__(self, crossovers):
        self.fig, self.axis = plt.subplots(3, 3, figsize=(12, 6), dpi=100)
        plt.ion()
        self.fig.show()
        self.median_evals = [[] for _ in range(len(crossovers))]
        self.success_rates = [[] for _ in range(len(crossovers))]
        self.fitness = [[] for _ in range(len(crossovers))]
        self.x = [[] for _ in range(len(crossovers))]
        self.y = [[] for _ in range(len(crossovers))]
        self.xlabels = []
        self.plot_titles = []

    def make_plots(self, inst_num, instance, crossovers, cx_num, median, success, fitness):
        self.median_evals[cx_num].append(median)
        self.success_rates[cx_num].append(success)
        self.fitness[cx_num].append(fitness)

        print(self.median_evals)

        self.x[cx_num].append(inst_num)
        self.y[cx_num] = [self.median_evals[cx_num], self.success_rates[cx_num], self.fitness[cx_num]]
        self.plot_titles = ["Median evaluations of " + crossovers[cx_num], "Success rate of " + crossovers[cx_num],
                            "Best fitness of " + crossovers[cx_num]]

        if cx_num == 0:
            self.xlabels.append(instance)

        self.fig.suptitle("Baseline results")

        for p in range(len(crossovers)):
            print(self.y[cx_num][p])
            self.axis[p, cx_num].bar(self.x[cx_num], self.y[cx_num][p], 0.8, color='green')
            self.axis[p, cx_num].set(xticks=np.arange(len(self.xlabels)), xticklabels=self.xlabels)
            self.axis[p, cx_num].tick_params(rotation=45)
            self.axis[p, cx_num].set_title(self.plot_titles[p])
        """
        self.axis[0, cx_num].bar(self.x[cx_num], self.median_evals[cx_num], 0.8, color='green')
        self.axis[0, cx_num].set(xticks=np.arange(len(self.xlabels)), xticklabels=self.xlabels)
        self.axis[0, cx_num].tick_params(rotation=45)
        self.axis[0, cx_num].set_title(self.plot_titles[0])
        self.axis[1, cx_num].bar(self.x[cx_num], self.success_rates[cx_num], 0.8, color='green')
        self.axis[1, cx_num].set(xticks=np.arange(len(self.xlabels)), xticklabels=self.xlabels)
        self.axis[1, cx_num].tick_params(rotation=45)
        self.axis[1, cx_num].set_title(self.plot_titles[1])
        self.axis[2, cx_num].bar(self.x[cx_num], self.fitness[cx_num], 0.8, color='green')
        self.axis[2, cx_num].set(xticks=np.arange(len(self.xlabels)), xticklabels=self.xlabels)
        self.axis[2, cx_num].tick_params(rotation=45)
        self.axis[2, cx_num].set_title(self.plot_titles[2])
        """

        plt.tight_layout()
        plt.draw()
        plt.pause(0.5)


if __name__ == "__main__":
    # settings:
    n = int(len(instances))
    crossovers = ["CustomCrossover", "UniformCrossover", "OnePointCrossover"]
    plot = True
    log_file = True

    # init:
    if plot:
        draw = PlotResults(crossovers)
    else:
        draw = None

    if log_file:
        for cx in crossovers:
            with open("output-{}.txt".format(cx), "w") as f:
                f.write("Layout: instance, population size, success rate, percentiles1, percentiles2, percentiles3, "
                        "best fitness \n")

    # main loop:
    for s in range(n):
        inst = instances[s, 2]
        for cx in crossovers:
            population_size = 2000
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

            if plot:
                draw.make_plots(s, instances[s, 0] + instances[s, 1], crossovers, crossovers.index(cx),
                                np.median(num_evaluations_list), num_success/num_runs, best_fitness)

            if log_file:
                with open("output-{}.txt".format(cx), "w") as f:
                    percentiles = np.percentile(num_evaluations_list, [10, 50, 90])
                    f.write("{} {} {} {} {} {}\n".format(instances[s, 0] + instances[s, 1], population_size,
                                                         num_success / num_runs, percentiles[0], percentiles[1],
                                                         percentiles[2], best_fitness))

    file_path = "./results.pdf"
    plt.savefig(file_path)
