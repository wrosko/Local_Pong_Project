from agents import TwoLayerAgent
import numpy as np


def create_random_agent():
    n_inputs = 2 + 2 + 4 + 4 + 1 + 1 # ball pos, ball vel, pad poss, pad vels, width, height. Total = 14
    n_hidden_neurons = 3
    
    hidden_weights = 2 * np.random.random((n_hidden_neurons, n_inputs)) - 1
    hidden_biases = 2 * np.random.random((n_hidden_neurons, 1)) - 1

    output_weights = 2 * np.random.random((1, n_hidden_neurons)) - 1
    output_bias = 2 * np.asarray(np.random.random()) - 1

    agent = TwoLayerAgent(hidden_weights, hidden_biases, output_weights, output_bias)

    return agent


def random_evaluation(agent):

    # Setup a random environment
    ball_position = 10.0 * np.random.random((2, 1))
    ball_velocity = np.random.random((2, 1))

    pad_positions = 5.0 * np.random.random((4, 1))
    pad_velocities = np.random.random((4, 1))

    court_width = 10.0
    court_height = 5.0

    n_evaluations = 1000

    agent.perceive_and_think(ball_position, ball_velocity, pad_positions, pad_velocities, court_width, court_height)
    return agent.desired_acceleration

if __name__ == "__main__":

    print "This script tests the evaluation time for the perceive_and_think() module of the agent. I.e., calling the neural network"
    
    n_agents = 10
    agents = [create_random_agent() for i in range(n_agents)]

    n_evaluations_per_agent = 1000
    
    import time
    tic = time.clock()

    acc = 0.0
    
    for agent in agents:
        for evaluation in range(n_evaluations_per_agent):
            acc += random_evaluation(agent)

    toc = time.clock()

    n_evaluations = n_agents * n_evaluations_per_agent

    average_acceleration = acc / float(n_evaluations)
    
    elapsed_time = toc - tic
    average_time = elapsed_time / float(n_evaluations)

    print "Elapsed time for {} evaluations = {} s".format(n_evaluations, elapsed_time)
    print "Average time = {} s, using {} random agents with {} evaluations each.".format(average_time, n_agents, n_evaluations_per_agent)
    print "Average acceleration = {}".format(average_acceleration)
