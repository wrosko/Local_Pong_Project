import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import game_constants

def update(num, ball_positions_list, pad_positions_list, ball_handle, pad_handles):
    bp = ball_positions_list[num, :]
    ball_handle.set_data(bp)
    
    offsets = 2 * [game_constants.pad_horizontal_offset] + 2 * [game_constants.court_width - game_constants.pad_horizontal_offset]
    pad_height = game_constants.pad_height

    pad_positions = pad_positions_list[num, :]
    for x, y, line in zip(offsets, pad_positions, pad_handles):
        line.set_data([x, x], [y, y + pad_height])

    return ball_handle,


def display(ball_position_list, pad_positions_list):
    
    fig1 = plt.figure()

    ball_handle, = plt.plot([], [], '*')
    pad_handles = [plt.plot([], [], '-')[0] for i in range(4)]
    
    plt.xlim(0, game_constants.court_width)
    plt.ylim(0, game_constants.court_height)
    interval = game_constants.frame_interval


    ball_ = animation.FuncAnimation(fig1,
                                    update,
                                    len(ball_position_list),
                                    fargs=(ball_position_list, pad_positions_list, ball_handle, pad_handles),
                                    interval=interval, blit=False, repeat = False)

    # To save the animation, use the command: line_ani.save('lines.mp4')
    # im_ani.save('im.mp4', metadata={'artist':'Guido'})

    plt.show()
