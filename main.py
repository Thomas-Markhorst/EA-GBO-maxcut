import numpy as np

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction

if __name__ == "__main__":
	crossovers = ["UniformCrossover", "CustomCrossover"]
	for cx in crossovers:
		inst = "maxcut-instances/setE/n0000160i05.txt"
		with open("output-{}.txt".format(cx),"w") as f:
			population_size = 10000
			num_evaluations_list = []
			num_runs = 5
			num_success = 0
			
			for i in range(num_runs):
				fitness = FitnessFunction.MaxCut(inst)
				genetic_algorithm = GeneticAlgorithm(fitness, population_size, variation=cx, evaluation_budget=400000, verbose=False)
				best_fitness, num_evaluations = genetic_algorithm.run()

				if best_fitness == fitness.value_to_reach:
					num_success += 1

				num_evaluations_list.append(num_evaluations)

			print("{}/{} runs successful".format(num_success, num_runs))
			print("{} evaluations (median)".format(np.median(num_evaluations_list)))
			percentiles = np.percentile(num_evaluations_list, [10, 50, 90])
			f.write("{} {} {} {} {}\n".format(population_size, num_success/num_runs, percentiles[0], percentiles[1], percentiles[2]))
