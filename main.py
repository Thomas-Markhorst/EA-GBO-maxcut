import numpy as np
import matplotlib.pyplot as plt
from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction

instances1 = np.array([
    ["Set A", "06i01", "maxcut-instances/setA/n0000006i01.txt"],
    ["Set A", "12i01", "maxcut-instances/setA/n0000012i01.txt"],
    ["Set A", "25i01", "maxcut-instances/setA/n0000025i01.txt"],
    ["Set A", "50i01", "maxcut-instances/setA/n0000050i01.txt"],
    ["Set A", "100i01", "maxcut-instances/setA/n0000100i01.txt"],
    ["Set B", "09i01", "maxcut-instances/setB/n0000009i01.txt"],
    ["Set B", "16i01", "maxcut-instances/setB/n0000016i01.txt"],
    ["Set B", "25i01", "maxcut-instances/setB/n0000025i01.txt"],
    ["Set B", "49i01", "maxcut-instances/setB/n0000049i01.txt"],
    ["Set B", "100i01", "maxcut-instances/setB/n0000100i01.txt"],
    ["Set C", "06i01", "maxcut-instances/setC/n0000006i01.txt"],
    ["Set C", "12i01", "maxcut-instances/setC/n0000012i01.txt"],
    ["Set C", "25i01", "maxcut-instances/setC/n0000025i01.txt"],
    ["Set C", "50i01", "maxcut-instances/setC/n0000050i01.txt"],
    ["Set C", "100i01", "maxcut-instances/setC/n0000100i01.txt"],
    ["Set D", "10i01", "maxcut-instances/setD/n0000010i01.txt"],
    ["Set D", "20i01", "maxcut-instances/setD/n0000020i01.txt"],
    ["Set D", "40i01", "maxcut-instances/setD/n0000040i01.txt"],
    ["Set D", "80i01", "maxcut-instances/setD/n0000080i01.txt"],
    ["Set D", "160i01", "maxcut-instances/setD/n0000160i01.txt"],
    ["Set E", "10i01", "maxcut-instances/setE/n0000010i01.txt"],
    ["Set E", "20i01", "maxcut-instances/setE/n0000020i01.txt"],
    ["Set E", "40i01", "maxcut-instances/setE/n0000040i01.txt"],
    ["Set E", "80i01", "maxcut-instances/setE/n0000080i01.txt"],
    ["Set E", "160i01", "maxcut-instances/setE/n0000160i01.txt"]
])

instances = np.array([
    ["Set A", "06i01", "maxcut-instances/setA/n0000006i01.txt"],
    ["Set B", "09i01", "maxcut-instances/setB/n0000009i01.txt"],
    ["Set C", "06i01", "maxcut-instances/setC/n0000006i01.txt"],
    ["Set D", "10i01", "maxcut-instances/setD/n0000010i01.txt"],
    ["Set E", "10i01", "maxcut-instances/setE/n0000010i01.txt"]
])


class PlotResults:

    def __init__(self, n, crossovers):
        self.fig, self.axis = plt.subplots(2, 3)
        plt.ion()
        self.fig.show()
        self.median_evals = np.zeros([n, len(crossovers)])
        self.success_rates = np.zeros([n, len(crossovers)])

    def make_plots(self, inst_num, instances, crossovers, cx_num, median, success):
        self.median_evals[inst_num, cx_num] = median
        self.success_rates[inst_num, cx_num] = success

        self.fig.suptitle("Baseline results")

        self.axis[0, cx_num].bar(np.arange(inst_num), self.median_evals[0:inst_num, cx_num], 0.8, color='green')
        self.axis[0, cx_num].set_title("Median evaluations of " + crossovers[cx_num])
        self.axis[1, cx_num].plot(self.success_rates[0:inst_num, cx_num], 'g')
        self.axis[1, cx_num].set_title("Success rate of " + crossovers[cx_num])

        plt.tight_layout()
        plt.draw()
        plt.pause(0.3)


if __name__ == "__main__":
    # settings:
    n = int(len(instances))
    crossovers = ["CustomCrossover", "UniformCrossover", "OnePointCrossover"]
    plot = True
    log_file = False

    if plot:
        draw = PlotResults(n, crossovers)

    for s in range(n):
        inst = instances[s, 2]
        for cx in crossovers:
            population_size = 500
            num_evaluations_list = []
            num_runs = 3
            num_success = 0

            for i in range(num_runs):
                fitness = FitnessFunction.MaxCut(inst)
                genetic_algorithm = GeneticAlgorithm(fitness, population_size, variation=cx, evaluation_budget=100000,
                                                     verbose=False)
                best_fitness, num_evaluations = genetic_algorithm.run()

                if best_fitness == fitness.value_to_reach:
                    num_success += 1

                num_evaluations_list.append(num_evaluations)

            if plot:
                draw.make_plots(s, inst, crossovers, crossovers.index(cx), np.median(num_evaluations_list), num_success/num_runs)

            if log_file:
                with open("output-{}.txt".format(cx), "w") as f:
                    print("{}/{} runs successful".format(num_success, num_runs))
                    print("{} evaluations (median)".format(np.median(num_evaluations_list)))
                    percentiles = np.percentile(num_evaluations_list, [10, 50, 90])
                    print(num_success / num_runs)
                    f.write("{} {} {} {} {}\n".format(population_size, num_success / num_runs, percentiles[0], percentiles[1], percentiles[2]))

    file_path = "./results.pdf"
    plt.savefig(file_path)

