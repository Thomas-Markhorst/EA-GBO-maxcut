import os
import numpy as np

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction

SET_LETTER = "A"

def get_txt_instances_of_set(setLetter: str):
  txt_set_file_names = []
  # Iterate directory
  for file in os.listdir(f"maxcut-instances/set{setLetter.upper()}/"):
    if file.endswith('.txt'):
      txt_set_file_names.append(f"maxcut-instances/set{setLetter.upper()}/{file}")

  return txt_set_file_names

if __name__ == "__main__":
  crossovers = ["UniformCrossover", "OnePointCrossover"]
  with open("metrics/results.txt", "w") as f:
    f.write("{} {} {} {} {} {} {} {} {}\n".format("set", "instance", "crossover", "population_size", "success_percentage", "evaluations_median", "percentile10", "percentile50", "percentile90"))

    for cx in crossovers:
      instances = get_txt_instances_of_set(SET_LETTER)
      # Only write metrics for instances 10-20 of a set as it takes way too much time to do them all
      for inst in instances[10:20]:
        population_size = 500
        num_evaluations_list = []
        num_runs = 30
        num_success = 0
        
        for i in range(num_runs):
          fitness = FitnessFunction.MaxCut(inst)
          genetic_algorithm = GeneticAlgorithm(fitness, population_size, variation=cx, evaluation_budget=100000, verbose=False)
          best_fitness, num_evaluations = genetic_algorithm.run()

          if best_fitness == fitness.value_to_reach:
            num_success += 1

          num_evaluations_list.append(num_evaluations)

        print("{}/{} runs successful".format(num_success, num_runs))
        print("{} evaluations (median)".format(np.median(num_evaluations_list)))
        percentiles = np.percentile(num_evaluations_list, [10, 50, 90])
        
        instance_name = inst[22:33]
        f.write("{} {} {} {} {} {} {} {} {}\n".format(SET_LETTER, instance_name, cx, population_size, num_success/num_runs, np.median(num_evaluations_list), percentiles[0], percentiles[1], percentiles[2]))