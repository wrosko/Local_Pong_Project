
if __name__ == "__main__":
    import numpy as np
    from agents import TwoLayerAgent
    from game import simulate_game
    from view import display
    from geneticalgorithm import decode_to_neural_network
    import geneticalgorithm as ga

    npzfile = np.load("BestChromosomes.npz")

    chromosomes = npzfile['best_chromosomes']
    agents = []

    for chromosome in chromosomes:
        input_weights,hidden_thresholds,hidden_weights,output_thresholds = decode_to_neural_network(chromosome, 14, 5, 1, 5, 5)
        
        agent = TwoLayerAgent(input_weights, hidden_thresholds, hidden_weights, output_thresholds)
        agents.append(agent)

    # Simulate a game. It returns the ID (0,1,2,3) of the agent who touched it, a ball data (position and velocity of every frame) and pad data (positions and velocities)
    last_pad_id, ball_data, pads_data, bounces = simulate_game(agents, True)

    # The ball_positions are the first elements in the ball_data, and so on. Extract it.
    ball_position_list = ball_data[0]
    pad_positions_list = pads_data[0]

    # Send these to the view-module for plotting
    display(ball_position_list, pad_positions_list)
