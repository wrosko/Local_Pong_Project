
if __name__ == "__main__":
    from agents import RandomAgent
    from game import simulate_game
    from view import display
    from example_two_layer_agent_think_test import create_random_agent as create_random_two_layer_agent

    # Initialize 4 instances of the RandomAgent class and store them in a list. These guys just accelerate at random (Every frame there's a 30 % probability to accelerate with any acceleration between -1 and 1)

    agents = [RandomAgent(0.3),
              create_random_two_layer_agent(),             
              RandomAgent(0.3),
              create_random_two_layer_agent()]              


    # Simulate a game. It returns the ID (0,1,2,3) of the agent who touched it, a ball data (position and velocity of every frame) and pad data (positions and velocities)
    last_pad_id, ball_data, pads_data, bounce_counter = simulate_game(agents, True)

    # The ball_positions are the first elements in the ball_data, and so on. Extract it.
    ball_position_list = ball_data[0]
    pad_positions_list = pads_data[0]


    print bounce_counter
    
    # Send these to the view-module for plotting
    display(ball_position_list, pad_positions_list)
