import numpy as np
import matplotlib.pyplot as plt


class InitPlots:

    def __init__(self):
        self.draw_bar = None
        self.draw_plot = None

    def initbar(self, bar_plot, crossovers, num_instances):
        if bar_plot:
            self.draw_bar = BarPlot(crossovers, num_instances)
        return self.draw_bar

    def initplot(self, pop_plot, crossovers, num_populations):
        if pop_plot:
            self.draw_plot = PopPlot(crossovers, num_populations)
        return self.draw_plot


class PopPlot:

    def __init__(self, crossovers, num_populations):
        self.fig, self.axis = plt.subplots(3, 1, figsize=(6, 12), dpi=100)
        plt.ion()
        self.fig.show()

        self.instance = ""
        self.crossovers = crossovers
        self.median_evals = [[] for _ in range(len(self.crossovers))]
        self.success_rates = [[] for _ in range(len(self.crossovers))]
        self.fitness = [[] for _ in range(len(self.crossovers))]
        self.num_populations = num_populations
        self.x = []
        self.y = [[] for _ in range(len(crossovers))]
        self.plot_titles = []
        self.colors = ["g", "b", "r"]

    def make_plots(self, instance, cx_num, median, success, fitness, population):
        self.instance = instance
        self.median_evals[cx_num].append(median)
        self.success_rates[cx_num].append(success)
        self.fitness[cx_num].append(fitness)

        self.y[cx_num] = [self.median_evals[cx_num], self.success_rates[cx_num], self.fitness[cx_num]]

        self.fig.suptitle("Instance {}: results for different population sizes".format(self.instance))
        self.plot_titles = ["Median evaluations of " + self.crossovers[cx_num], "Success rate of " +
                            self.crossovers[cx_num], "Best fitness of " + self.crossovers[cx_num]]

        if cx_num == len(self.crossovers)-1:
            self.x.append(population)
            for j in range(3):
                for r in range(len(self.crossovers)):
                    self.axis[j].plot(self.x, self.y[r][j], self.colors[r], label=self.crossovers[r])
                    self.axis[j].set_title(self.plot_titles[j])
                    if len(self.x) < 2:
                        self.axis[j].legend()
                        self.axis[j].set_xlabel("Population size")

            plt.tight_layout()
            plt.draw()
            plt.pause(0.5)

            if len(self.x) == self.num_populations:
                self.reset_save_plot()

    def reset_save_plot(self):
        plt.savefig("./plots/pop_plot_{}.pdf".format(self.instance))

        # self.fig, self.axis = plt.subplots(3, 1, figsize=(6, 12), dpi=100)
        # plt.ion()
        # self.fig.show()

        # self.median_evals = [[] for _ in range(len(self.crossovers))]
        # self.success_rates = [[] for _ in range(len(self.crossovers))]
        # self.fitness = [[] for _ in range(len(self.crossovers))]
        # self.x = []


class BarPlot:

    def __init__(self, crossovers, num_instances):
        self.fig, self.axis = plt.subplots(3, 1, figsize=(6, 14), dpi=100)
        plt.ion()
        self.fig.show()
        self.median_evals = [[] for _ in range(len(crossovers))]
        self.success_rates = [[] for _ in range(len(crossovers))]
        self.fitness = [[] for _ in range(len(crossovers))]
        self.x = []
        self.y = [[] for _ in range(len(crossovers))]
        self.width = 0.3
        self.width_scales = [-self.width, 0, self.width]
        self.colors = ["green", "blue", "yellow"]
        self.xlabels = []
        self.plot_titles = []
        self.num_instances = num_instances
        self.crossovers = crossovers

    def make_bar_plots(self, inst_num, instance, cx_num, median, success, fitness):
        self.median_evals[cx_num].append(median)
        self.success_rates[cx_num].append(success)
        self.fitness[cx_num].append(fitness)

        print(self.median_evals)

        if cx_num == 0:
            self.xlabels.append(instance)

        self.x = np.arange(len(self.xlabels))
        print((self.x - (3 / 2) * self.width))
        print(self.median_evals[0])
        self.y[cx_num] = [np.log(self.median_evals[cx_num]), self.success_rates[cx_num], np.log(self.fitness[cx_num])]
        self.plot_titles = ["log median evaluations of " + self.crossovers[cx_num], "Success rate of " + self.crossovers[cx_num],
                            "Log best fitness of " + self.crossovers[cx_num]]

        self.fig.suptitle("Baseline results")

        for m in range(3):
            self.axis[m].bar(self.x + self.width_scales[cx_num], self.y[cx_num][m], self.width,
                         color=self.colors[cx_num], label=self.crossovers[cx_num])
            self.axis[m].set(xticks=self.x, xticklabels=self.xlabels)
            self.axis[m].tick_params(rotation=45)
            self.axis[m].set_title(self.plot_titles[m])
            if inst_num < 1:
                self.axis[m].legend()

        plt.tight_layout()
        plt.draw()
        plt.pause(0.5)

        if inst_num+1 == self.num_instances:
            plt.savefig("./plots/bar_plot.pdf")

