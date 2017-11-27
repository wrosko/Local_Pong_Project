#from __future__ import print_function
import math
import numpy as np
from agents import TwoLayerAgent
from game import simulate_game

def cross(ch1,ch2):
    num_genes = len(ch1);
    crossover_point = math.floor(np.random.random_sample()*num_genes)
    ch1[-crossover_point:],ch2[-crossover_point:] = (
            ch2[-crossover_point:],ch1[-crossover_point:])
    return(ch1,ch2)

def decode_chromosome(chromosome,num_variables,variable_range):
    num_genes = len(chromosome);
    num_per_var = int(math.floor(num_genes/num_variables))
    x = [0]*num_variables;

    for i_var in range(0,num_variables):
        for i_gene in range(0,num_per_var):
            x[i_var] += chromosome[i_gene+num_per_var*i_var]*2.0**(-i_gene-1)
        x[i_var] = (
                -variable_range + 
                2.0*variable_range*x[i_var]/(1.0 - 2.0**(-num_per_var)))
    return x

def decode_to_neural_network(chromosome,num_input_neurons,num_hidden_neurons,
                             num_output_neurons,weight_range,threshold_range):
    
    num_input_weights = num_hidden_neurons * num_input_neurons
    num_hidden_thresholds = num_hidden_neurons
    
    num_hidden_weights = num_output_neurons * num_hidden_neurons
    num_output_thresholds = num_output_neurons
    
    num_genes = len(chromosome)
    num_variables = (num_input_weights 
                     + num_hidden_thresholds 
                     + num_hidden_weights 
                     + num_output_thresholds)
    num_genes_per_value = math.floor(num_genes/num_variables)
    
    start_gene = 0
    end_gene = num_input_weights * num_genes_per_value
    tmp_input_weights = (
            decode_chromosome(
                    chromosome[start_gene:end_gene],num_input_weights,
                    weight_range))
    input_weights = np.reshape(tmp_input_weights,
                               (num_hidden_neurons,num_input_neurons))
    
    start_gene = end_gene
    end_gene += num_hidden_thresholds * num_genes_per_value
    hidden_thresholds = (
            decode_chromosome(
                    chromosome[start_gene:end_gene],num_hidden_thresholds,
                    threshold_range))

    hidden_thresholds = np.reshape(hidden_thresholds, (num_hidden_neurons, 1))
    
    start_gene = end_gene
    end_gene += num_hidden_weights * num_genes_per_value
    tmp_hidden_weights = decode_chromosome(chromosome[start_gene:end_gene],
                                           num_hidden_weights,weight_range)
    hidden_weights = np.reshape(tmp_hidden_weights,
                                (num_output_neurons,num_hidden_neurons))
    
    start_gene = end_gene
    end_gene += num_output_thresholds * num_genes_per_value
    output_thresholds = decode_chromosome(
            chromosome[start_gene:end_gene],num_output_thresholds,
            threshold_range)
    
    return (input_weights, hidden_thresholds, hidden_weights, output_thresholds)
    

def evaluate_populations(generation, number_of_matchups, number_of_games_per_matchup, populations, agent_generating_functions, score_functions):

    number_of_populations = len(populations) # a.k.a number of agents
    population_size = len(populations[0]) # Size of an individual population
    score_card = np.zeros(number_of_games_per_matchup) # Keeps track of winning agent for a given matchup
    last_ball_position_list = np.zeros((number_of_games_per_matchup, 2))
    last_pad_centers_list = np.zeros((number_of_games_per_matchup, 4))
    bounce_counter_list = np.zeros((number_of_games_per_matchup, 4))    

    agents = [None] * number_of_populations 

    scores = np.zeros((number_of_populations, population_size))
    games = np.zeros((number_of_populations, population_size))
    
    for matchup_counter in range(number_of_matchups):
        # Select random participants from individuals
        participants = np.random.randint(0, population_size, number_of_populations)

        # Generate agents, i.e., create a match-up
        for population_index in range(number_of_populations):
            population = populations[population_index]
            participant = participants[population_index]

            chromosome = population[participant] # Select the chromosome of current participant

            # Generate an agent using the chromosome
            agents[population_index] = agent_generating_functions[population_index](chromosome)

        # Play a number of games for this match-up
        for game_counter in range(number_of_games_per_matchup):
            scoring_agent, last_ball_position, last_pad_centers, bounce_counter = simulate_game(agents, False)

            score_card[game_counter] = scoring_agent
            last_ball_position_list[game_counter, :] = last_ball_position
            last_pad_centers_list[game_counter, :] = last_pad_centers
            bounce_counter_list[game_counter, :] = bounce_counter
    
        # For every participating participant, evaluate its fitness
        for population_index in range(number_of_populations):
            individual = participants[population_index]
            score_function = score_functions[population_index]

            # Increase score of this guy and number of games played
            scores[population_index, individual] += score_function(generation, score_card, last_ball_position_list, last_pad_centers_list, bounce_counter_list)
            games[population_index, individual] += number_of_games_per_matchup

    # Fitness = score per game
    games[games == 0.0] = 1.0
    fitnesses = scores / games
    
    return fitnesses


def initialize_populations(num_populations, population_size,num_genes):
    population = (np.random.rand(num_populations, population_size, num_genes) > 0.5).astype(int)
    return population

def insert_best_individual(population,best_individual,num_best_to_insert):
    for i in range(num_best_to_insert):
        population[i] = best_individual
    return population
        
def mutate(chromosome,mutation_probability):
        for gene in chromosome:
            if np.random.random_sample() < mutation_probability:
                gene = 1-gene;
        return chromosome
    
def perform_one_evolution_timestep(
        population,fitness,TOURNAMENT_SELECTION_PARAMETER,TOURNAMENT_SIZE,
        CROSSOVER_PROBABILITY,MUTATION_PROBABILITY,best_chromosome,
        NUM_BEST_TO_INSERT):
    
    POPULATION_SIZE = len(population);
    new_population = np.zeros(population.shape)
    for individual_index in range(0,POPULATION_SIZE,2):
            ind1 = tournament_select(
                    fitness,TOURNAMENT_SELECTION_PARAMETER,TOURNAMENT_SIZE)
            ind2 = tournament_select(
                    fitness,TOURNAMENT_SELECTION_PARAMETER,TOURNAMENT_SIZE)
            chromosome1 = population[ind1]
            chromosome2 = population[ind2]
        
            if np.random.random_sample() < CROSSOVER_PROBABILITY:
                chromosome1,chromosome2 =  cross(chromosome1,chromosome2)
        
            new_population[individual_index] =  chromosome1
            new_population[individual_index+1] = chromosome2          
    
        # Mutate the new population
    for newChromosome in new_population:
        newChromosome = mutate(newChromosome,MUTATION_PROBABILITY)
    
        # Elitism
    new_population = insert_best_individual(new_population,best_chromosome,NUM_BEST_TO_INSERT)
    
    return new_population
    
def tournament_select(fitness,tournament_selection_parameter,tournament_size):
    # fitness values in a numpy array
    selected_individuals = np.random.choice(len(fitness),tournament_size)
    selected_fitness_values = fitness[selected_individuals]
    perm = selected_fitness_values.argsort();
    sorted_individuals = selected_individuals[perm[::-1]]
    
    # choose best one with probability 
    for individual in sorted_individuals:
        if np.random.random_sample() < tournament_selection_parameter:
            return individual
    return sorted_individuals[-1]


def train_individuals(number_of_games_per_matchup, number_of_matchups_per_generation,
                      number_of_generations, population_size, number_of_genes,
                      tournament_selection_parameter, tournament_size,
                      crossover_probability, mutation_probability,
                      number_of_best_to_insert,
                      agent_generating_functions, score_functions):

    number_of_populations = len(agent_generating_functions)
    assert number_of_populations == len(score_functions)
    
    populations = initialize_populations(number_of_populations, population_size, number_of_genes)
    population_fitnesses = np.zeros((number_of_populations, population_size))
    
    def get_best_individual(population, population_fitness):
        best_index = np.argmax(population_fitness)
        fitness = population_fitness[best_index]
        chromosome = population[best_index]
        return (chromosome, fitness)
    
    for generation_counter in range(number_of_generations):

        # Play games and evaluate fitnesses
        population_fitnesses = evaluate_populations(generation_counter,
                                                    number_of_matchups_per_generation,
                                                    number_of_games_per_matchup,
                                                    populations,
                                                    agent_generating_functions,
                                                    score_functions)

        # Evolve the populations one by one

        print("Generation {}".format(generation_counter))
        
        for population, population_fitness in zip(populations, population_fitnesses):
            best_chromosome, best_fitness = get_best_individual(population, population_fitness)
            #print (population_fitness)
            print("Best fitness = {}".format(best_fitness))
            
            population = perform_one_evolution_timestep(population, population_fitness,
                                                        tournament_selection_parameter, tournament_size,
                                                        crossover_probability, mutation_probability,
                                                        best_chromosome,
                                                        number_of_best_to_insert)
        print("--------------------")


    # Now that we're done evolving, find the best guys.
    best_chromosomes = []
    for population, population_fitness in zip(populations, population_fitnesses):
        best_chromosome, best_fitness = get_best_individual(population, population_fitness)
        best_chromosomes.append(best_chromosome)
        
    return best_chromosomes
    
