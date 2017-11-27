import geneticalgorithm as ga
import numpy as np

def score_general(generation, score_card, last_ball_position_list, last_pad_centers_list, bounce_counter_list, team, individual, selfishness):
    score = 0.0

    if generation > 100:
    
        for scoring_agent in score_card:
            if scoring_agent in team:
                # Update team score
                score += 1.0 - selfishness

                # Update individual score
                if scoring_agent == individual:
                    score += selfishness
    else:
        for ball_position, pad_centers, bounces in zip(last_ball_position_list,
                                                       last_pad_centers_list,
                                                       bounce_counter_list):

            pad_center = pad_centers[individual]
            score += 1 / np.linalg.norm(ball_position[1] - pad_center)

    return score

if __name__ == "__main__":

    # Neural network
    NUM_HIDDEN_NEURONS = 4
    NUM_OUTPUT_NEURONS = 1;

    # Game
    NUM_BALLS = 1;
    NUM_PLAYERS_PER_TEAM = 2
    
    # Genetic algorithm
    NUMBER_OF_GENERATIONS = 10
    NUMBER_OF_GAMES_PER_MATCHUP = 100
    NUMBER_OF_MATCHUPS_PER_GENERATION = 10
    POPULATION_SIZE = 100
    
    NUMBER_OF_BEST_TO_INSERT = 1
    NUM_GENES_PER_VAR = 10;
    
    WEIGHT_RANGE = 5.0
    THRESHOLD_RANGE = 5.0
    CROSSOVER_PROBABILITY = 0.8
    TOURNAMENT_SIZE = 2
    TOURNAMENT_SELECTION_PARAMETER = 0.8

    # Derived parameters
    num_input_neurons = 2*NUM_PLAYERS_PER_TEAM*2 + NUM_BALLS*4+2    
    num_variables = (NUM_HIDDEN_NEURONS * (num_input_neurons + 1) 
            + NUM_OUTPUT_NEURONS * (NUM_HIDDEN_NEURONS + 1))
    
    num_genes = num_variables * NUM_GENES_PER_VAR    
    mutation_probability = 2.0/float(num_genes)

    # Score functions
    score_functions = [lambda generation, score_card, last_ball_position, last_pad_positions, bounce_counter: score_general(generation, score_card, last_ball_position, last_pad_positions, bounce_counter, team = [0, 1], individual = 0, selfishness = 1.0),
                       lambda generation, score_card, last_ball_position, last_pad_positions, bounce_counter: score_general(generation, score_card, last_ball_position, last_pad_positions, bounce_counter, team = [0, 1], individual = 1, selfishness = 1.0),
                       lambda generation, score_card, last_ball_position, last_pad_positions, bounce_counter: score_general(generation, score_card, last_ball_position, last_pad_positions, bounce_counter, team = [2, 3], individual = 2, selfishness = 0.0),
                       lambda generation, score_card, last_ball_position, last_pad_positions, bounce_counter: score_general(generation, score_card, last_ball_position, last_pad_positions, bounce_counter, team = [2, 3], individual = 3, selfishness = 0.0)]

    # Agent generating functions
    def agent_generating_function(chromosome):
        from geneticalgorithm import decode_to_neural_network as decode
        from agents import TwoLayerAgent
                       
        hidden_weights, hidden_thresholds, output_weights, output_thresholds = decode(chromosome,
                                                                                      num_input_neurons,
                                                                                      NUM_HIDDEN_NEURONS,
                                                                                      NUM_OUTPUT_NEURONS,
                                                                                      WEIGHT_RANGE,
                                                                                      THRESHOLD_RANGE)

        agent = TwoLayerAgent(hidden_weights, hidden_thresholds, output_weights, output_thresholds)
        return agent

    agent_generating_functions = [agent_generating_function] * 4

                       
    best_chromosomes = ga.train_individuals(NUMBER_OF_GAMES_PER_MATCHUP, NUMBER_OF_MATCHUPS_PER_GENERATION,
                                            NUMBER_OF_GENERATIONS, POPULATION_SIZE, num_genes,
                                            TOURNAMENT_SELECTION_PARAMETER, TOURNAMENT_SIZE,
                                            CROSSOVER_PROBABILITY, mutation_probability,
                                            NUMBER_OF_BEST_TO_INSERT,
                                            agent_generating_functions, score_functions)
                                            
    np.savez("BestChromosomes", best_chromosomes = best_chromosomes)
