

from example_two_layer_agent_think_test import create_random_agent

if __name__ == "__main__":
    from game import simulate_game
    from agents import RandomAgent

    # Init agents
    agents = [create_random_agent() for i in range(4)]

    n_games = 1000
    agent_score = [0, 0, 0, 0]

    import time
    tic = time.clock()

    # Run a lot of games, save only the winners
    for game_counter in range(n_games):
        last_pad_id = simulate_game(agents, False)

        if last_pad_id != None: # If not a draw.
            agent_score[last_pad_id] += 1

        if game_counter % 100 == 0:
            print game_counter
            
    toc = time.clock()

    # Calculate the score of the teams
    team_one_score = sum(agent_score[:2])
    team_two_score = sum(agent_score[2:])

    print "{} games played. Duration: {}. Average time per game = {}".format(n_games, toc - tic, n_games / float(toc - tic) / 1000.0)
    print "Team one scores, agent 1: {} agent 2: {}, total: {}".format(agent_score[0],
                                                                       agent_score[1],
                                                                       team_one_score)
    
    print "Team two scores, agent 1: {} agent 2: {}, total: {}".format(agent_score[2],
                                                                       agent_score[3],
                                                                       team_two_score)

