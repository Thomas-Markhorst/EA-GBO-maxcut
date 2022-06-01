import numpy as np

from FitnessFunction import FitnessFunction, MaxCut
from Individual import Individual


def uniform_crossover(individual_a: Individual, individual_b: Individual, p=0.5):
    assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
    l = len(individual_a.genotype)
    offspring_a = Individual(l)
    offspring_b = Individual(l)

    m = np.random.choice((0, 1), p=(p, 1 - p), size=l)
    offspring_a.genotype = np.where(m, individual_a.genotype, individual_b.genotype)
    offspring_b.genotype = np.where(1 - m, individual_a.genotype, individual_b.genotype)

    return [offspring_a, offspring_b]


def one_point_crossover(individual_a: Individual, individual_b: Individual):
    assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
    l = len(individual_a.genotype)
    offspring_a = Individual(l)
    offspring_b = Individual(l)

    l = len(individual_a.genotype)
    m = np.arange(l) < np.random.randint(l + 1)
    offspring_a.genotype = np.where(m, individual_a.genotype, individual_b.genotype)
    offspring_b.genotype = np.where(~m, individual_a.genotype, individual_b.genotype)

    return [offspring_a, offspring_b]


def two_point_crossover(individual_a: Individual, individual_b: Individual):
    assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
    offspring_a = Individual()
    offspring_b = Individual()

    l = len(individual_a.genotype)
    m = (np.arange(l) < np.random.randint(l + 1)) ^ (np.arange(l) < np.random.randint(l + 1))
    offspring_a.genotype = np.where(m, individual_b.genotype, individual_a.genotype)
    offspring_b.genotype = np.where(~m, individual_b.genotype, individual_a.genotype)

    return [offspring_a, offspring_b]


def custom_crossover(fitness: FitnessFunction, individual_a: Individual, individual_b: Individual):
    assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
    l = len(individual_a.genotype)
    offspring_a = Individual(l)
    offspring_b = Individual(l)

    # Implement your custom crossover here
    offspring_a.genotype = individual_a.genotype.copy()
    offspring_b.genotype = individual_b.genotype.copy()
    if isinstance(fitness, MaxCut):
        # select a clique in the chain where we do the crossover
        clique_index_to_split = np.random.choice(len(fitness.cliques) - 1) + 1

        m = np.zeros(l)
        # select which nodes should be swapped with other individual
        for clique in fitness.cliques[:clique_index_to_split]:
            m[list(clique)] = 1
        # combine the two genes
        offspring_a.genotype = np.where(m, individual_a.genotype, individual_b.genotype)
        offspring_b.genotype = np.where(1 - m, individual_a.genotype, individual_b.genotype)

    return [offspring_a, offspring_b]


def get_suitable_indices(fitness: FitnessFunction, genotype_length):
    if isinstance(fitness, MaxCut):
        # we determine which vertices connect the cliques, for set D
        sorted_adjacency = sorted(fitness.adjacency_list, key=lambda x: len(fitness.adjacency_list[x]), reverse=True)
        last_size = len(fitness.adjacency_list[sorted_adjacency[0]])
        indices = [sorted_adjacency[0]]
        for index in sorted_adjacency[1:]:
            if len(fitness.adjacency_list[index]) == last_size:
                indices.append(index)
            else:
                break

        return indices
