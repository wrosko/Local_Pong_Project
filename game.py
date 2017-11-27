import numpy as np # Importing numpy, giving it the shorthand np for convenience
import matplotlib.pyplot as plt # For plotting
import matplotlib.animation as animation # For animation plotting
import game_constants 
from agents import Agent

def _check_and_adjust_for_vertical_bounce(old_time, old_x, old_y, old_vx, old_vy, pred_time, pred_x, pred_y, pred_vx, pred_vy, court_height):
    # In order to not duplicate code (because the pads need to be checked later), we will check and adjust in a general function. 
    # For convenience, the general function works by transforming the ball coords to a system attached to the top/bottom.
    # In this system it checks the condition for a bounce (for walls, its just that we go outside)
    # When a bounce is detected, it will adjust the time, position to point at the bounce-location.
    # As such, we pass the transformation functions, condition function and adjustment functions to the general function.

    # Create the coordinate transformations    
    x_transform = lambda x: x - old_x
    x_transforms = [x_transform, ] * 2
    
    x_inv_transform = lambda x_: x_ + old_x
    x_inv_transforms = [x_inv_transform, ] * 2
    
    y_transforms = [lambda y: y,
                    lambda y: court_height - y]

    y_inv_transforms = [lambda y_: y_,
                        lambda y_: court_height - y_]

    # Set up the condition function
    condition_function = lambda old_x_, old_y_, pred_x_, pred_y_: old_y_ * pred_y_ < 0 # Signs changed, means we crossed.

    # Set up the adjustment function
    def adjustment_function(old_time, old_x_, old_y_, old_vx_, old_vy_):
        # Calculate intersection time and position
        new_time = old_time - old_y_ / float(old_vy_)
            
        new_x_ = old_x_ + (new_time - old_time) * old_vx_
        new_y_ = 0.0
            
        # Assign velocity
        new_vx_ = old_vx_
        new_vy_ = - old_vy_ # Flip this one

        return new_time, new_x_, new_y_, new_vx_, new_vy_    
        
    return _check_and_adjust_for_bounce(old_time, old_x, old_y, old_vx, old_vy,
                                        pred_time, pred_x, pred_y, pred_vx, pred_vy,
                                        x_transforms, x_inv_transforms,
                                        y_transforms, y_inv_transforms,
                                        condition_function,
                                        adjustment_function)[:-1] # All except the last is needed.

def _check_and_adjust_for_pad_bounce(old_time, old_x, old_y, old_vx, old_vy, pred_time, pred_x, pred_y, pred_vx, pred_vy, pad_positions, pad_horizontal_offset, pad_height, court_width):

    # As in the case with the top/bottom bounce, we will pass the necessary transformations, condition and adjustment functions to the general bounce function. Set them up.

    # The x-transformations
    x_transforms = [lambda x: -x + pad_horizontal_offset, # Transform for coordinates on left pad line
                    lambda x: -x + pad_horizontal_offset, # Two left pads
                    lambda x: x - (court_width - pad_horizontal_offset), # Transform for right pad line
                    lambda x: x - (court_width - pad_horizontal_offset)] # Transform for right pad line

    x_inv_transforms = [lambda x_: -x_ + pad_horizontal_offset,
                        lambda x_: -x_ + pad_horizontal_offset,                             
                        lambda x_: x_ + (court_width - pad_horizontal_offset),
                        lambda x_: x_ + (court_width - pad_horizontal_offset)]

    # The y-transformations
    y_transforms = [lambda y, pp = pp: y - pp for pp in pad_positions] # NOTICE: lock in pp by pp = pp, otherwise last value of pp is used.
    y_inv_transforms = [(lambda y, pp = pp: y + pp) for pp in pad_positions]
    
    # The bounce conditions (they're all the same, since they will be passed a )
    sign_change_function = lambda old_x_, pred_x_: old_x_ * pred_x_ < 0 # Signs changed, means we crossed.
    condition_function = lambda old_x_, old_y_, pred_x_, pred_y_: sign_change_function(old_x_, pred_x_) and 0 <= (old_y_ + pred_y_) / 2.0 <= pad_height

    def adjustment_function(old_time, old_x_, old_y_, old_vx_, old_vy_):
        # Calculate intersection time and position
        new_time = old_time - old_x_ / float(old_vx_)

        new_x_ = 0.0
        new_y_ = old_y_ + (new_time - old_time) * old_vy_

        # Calculate an angle
        pad_angle_max = game_constants.pad_angle_max
        pad_angle_min = game_constants.pad_angle_min
        k = (pad_angle_max - pad_angle_min) / float(pad_height)
        theta = (k * new_y_ + pad_angle_min) * (np.pi / 180.0)
        
        # Assign velocity
        new_vx_ = np.cos(theta)
        new_vy_ = -np.sin(theta)

        return new_time, new_x_, new_y_, new_vx_, new_vy_    

    return _check_and_adjust_for_bounce(old_time, old_x, old_y, old_vx, old_vy,
                                        pred_time, pred_x, pred_y, pred_vx, pred_vy,
                                        x_transforms, x_inv_transforms,
                                        y_transforms, y_inv_transforms,
                                        condition_function,
                                        adjustment_function)


def _check_and_adjust_for_bounce(old_time, old_x, old_y, old_vx, old_vy, pred_time, pred_x, pred_y, pred_vx, pred_vy, x_transforms, x_inv_transforms, y_transforms, y_inv_transforms, condition_function, adjustment_function):
    
    bounced = False # whether we bounced or not. If we bounced, we will adjust the time, positions and velocities and pass it back to the update function. The update function will then run again from the point where the bounce occured.
    bounce_index = None

    # Deduce velocity transformations from the position transformations
    vx_transforms = [lambda x, xtf = xtf: xtf(x) - xtf(0) for xtf in x_transforms]
    vx_inv_transforms = [lambda x_, x_itf = x_itf: x_itf(x_) - x_itf(0) for x_itf in x_inv_transforms]
    
    vy_transforms = [lambda y, ytf = ytf: ytf(y) - ytf(0) for ytf in y_transforms]
    vy_inv_transforms = [lambda y_, y_itf = y_itf: y_itf(y_) - y_itf(0) for y_itf in y_inv_transforms]

    indices = range(len(x_transforms)) # We enumerate stuff so we can identify with what the collision occured (we really only care about the pad indices)

    # Loop over all things that we can bounce over (this function is called with the top and bottom or the pads as arguments). So we loop over everything we need to attach a coordinate system to any of these.
    for index, x_tf, x_itf, y_tf, y_itf, vx_tf, vx_itf, vy_tf, vy_itf in zip(indices,
                                                                             x_transforms, x_inv_transforms,
                                                                             y_transforms, y_inv_transforms,
                                                                             vx_transforms, vx_inv_transforms,
                                                                             vy_transforms, vy_inv_transforms):
        
        # Transform coordinates to a coordinate system attached to whatever we might bounce against
        old_x_ = x_tf(old_x); old_y_ = y_tf(old_y)
        old_vx_ = vx_tf(old_vx); old_vy_ = vy_tf(old_vy)
        pred_x_ = x_tf(pred_x); pred_y_ = y_tf(pred_y)
        pred_vx_ = vx_tf(pred_vx); pred_vy_ = vy_tf(pred_vy)

        # Check the bounce condition. Did we cross top/bottom or perhaps a pad?
        would_bounce = condition_function(old_x_, old_y_, pred_x_, pred_y_)

        if would_bounce:
            # If we should have bounced here, we adjust to the bounce position.
            
            bounce_index = index # Record which item we bounced with
            new_time, new_x_, new_y_, new_vx_, new_vy_ = adjustment_function(old_time, old_x_, old_y_, old_vx_, old_vy_) # Adjust the positions and velocities as necessary
            
            # Transform back.
            new_x = x_itf(new_x_); new_y = y_itf(new_y_)
            new_vx = vx_itf(new_vx_); new_vy = vy_itf(new_vy_)
            break # Can't bounce twice anyway.
            
    if not would_bounce:
        # If there wasn't a bounce we just return the predicted values
        new_time = pred_time
        new_x = pred_x; new_y = pred_y
        new_vx = pred_vx; new_vy = pred_vy

    return new_time, new_x, new_y, new_vx, new_vy, bounce_index


def _update_ball(ball_position, ball_velocity, last_pad_id, pad_positions, pad_horizontal_offset, pad_height, court_width, court_height, old_time, target_time):
    old_x, old_y = ball_position
    old_vx, old_vy = ball_velocity

    # Predict where the ball would be in spacetime without interference
    pred_time = target_time # If everything goes smoothly, we simply get to the target time
    pred_x, pred_y = ball_position + ball_velocity * (target_time - old_time)
    pred_vx, pred_vy = ball_velocity # We predict this to remain the same.

    # Now we check if there was a collision, because then our prediction is wrong.
    # So we find the point of collision and try again from there.

    new_time, new_x, new_y, new_vx, new_vy = _check_and_adjust_for_vertical_bounce(old_time, old_x, old_y, old_vx, old_vy, pred_time, pred_x, pred_y, pred_vx, pred_vy, court_height)
    
    vertical_bounce = new_time < pred_time # If we were stopped in our tracks, there was a collision.
    
    if not vertical_bounce:
        # If we didn't bounce vertically, we are good to check for the pad bounce.
        new_time, new_x, new_y, new_vx, new_vy, bounce_index = _check_and_adjust_for_pad_bounce(old_time, old_x, old_y, old_vx, old_vy, pred_time, pred_x, pred_y, pred_vx, pred_vy, pad_positions, pad_horizontal_offset, pad_height, court_width)
        
        if bounce_index != None:
            last_pad_id = bounce_index
            
        pad_bounce = new_time < pred_time 
    
    # The positions are set anew (if there were no collisions, these are just the predicted values)
    ball_position[0] = new_x
    ball_position[1] = new_y

    ball_velocity[0] = new_vx
    ball_velocity[1] = new_vy

    if vertical_bounce or pad_bounce:
        # We bounced. This means we have only updated the ball trajectory partially. Proceed from the new position and time.
        return _update_ball(ball_position, ball_velocity, last_pad_id, pad_positions, pad_horizontal_offset, pad_height, court_width, court_height, new_time, target_time)
    else:
        # We reached the targetted time. Simply return the values.
        return ball_position, ball_velocity, last_pad_id


def _update_pads(pad_positions, pad_velocities, pad_accelerations, pad_height, court_height, delta_time):
    # Update pad positions
    pad_velocities += pad_accelerations * delta_time
    pad_positions += pad_velocities * delta_time

    # Ensure pads didn't move outside of court

    for index, pad_position in enumerate(pad_positions):
        halt_pad = False
        
        if pad_position + pad_height > court_height:
            pad_positions[index] = court_height - pad_height
            halt_pad = True
        elif pad_position  < 0:
            pad_positions[index] = 0.0
            halt_pad = True

        if halt_pad:
            pad_velocities[:] = 0.0
            pad_accelerations[:] = 0.0

    return pad_positions, pad_velocities, pad_accelerations


def _check_for_game_over(ball_new_x, court_width, pad_horizontal_offset):
    return ball_new_x < pad_horizontal_offset or ball_new_x > court_width - pad_horizontal_offset


def _time_evolution(time, ball_position, ball_velocity, last_pad_id, pad_positions, pad_velocities, pad_accelerations, pad_horizontal_offset, pad_height, court_width, court_height, delta_time = 1.0): # Function with default values

    # Throw an error if a ball is given an initial position outside the court
    assert 0 <= ball_position[0] <= court_width
    assert 0 <= ball_position[1] <= court_height
    
    # Throw an error if a pad is outside the court
    for pos in pad_positions:
        assert 0 <= pos and pos + pad_height <= court_height

    ball_position, ball_velocity, last_pad_id = _update_ball(ball_position, ball_velocity, last_pad_id, pad_positions, pad_horizontal_offset, pad_height, court_width, court_height, time, time + delta_time)
    
    # Check if we went outside horizontally, i.e., if the game is over.                
    game_over = _check_for_game_over(ball_position[0], court_width, pad_horizontal_offset)

    pad_positions, pad_velocities, pad_accelerations = _update_pads(pad_positions, pad_velocities, pad_accelerations, pad_height, court_height, delta_time)
    
    time += delta_time
    return game_over, last_pad_id, time, ball_position, ball_velocity, pad_positions, pad_velocities, pad_accelerations


def _random_velocity(min_angle, max_angle):
    angle = np.random.uniform(min_angle, max_angle)

    if np.random.random() < 0.5:
        # Rotate 180 degrees
        angle += 180

    angle *= np.pi / 180
    velocity = np.asarray([np.cos(angle), np.sin(angle)])
    return velocity
        

def simulate_game(agents, save_time_series = True):
    assert len(agents) == 4

    # How big a time step is
    delta_time = game_constants.delta_time
    
    # Define court dimensions
    court_width = game_constants.court_width
    court_height = game_constants.court_height

    # Pad properties
    pad_horizontal_offset = game_constants.pad_horizontal_offset # How far left/right from edges the pads are
    pad_height = game_constants.pad_height

    # Initialize ball position
    ball_position = np.asarray([court_width / 2.0, court_height / 2.0]).reshape(2, 1)

    # Initialize random direction of unit velocity
    ball_velocity = _random_velocity(game_constants.start_angle_min, game_constants.start_angle_max).reshape(2, 1)

    # Initialize pad positions
    y1 = court_height / 3.0 - pad_height / 2.0
    y2 = 2.0 * court_height / 3.0 - pad_height / 2.0
    pad_positions = np.asarray([y1, y2, y1, y2]).reshape(4, 1)

    # Initialize velocities and accelerations to zero
    pad_velocities = np.zeros(4).reshape(4, 1)
    pad_accelerations = np.zeros(4).reshape(4, 1)
    
    game_over = False

    frames_shown = 0
    #print frames_shown, ball_position, pad_positions

    if save_time_series:
        # Preallocate arrays. This allows for faster appending
        max_frames = game_constants.max_frames
        ball_position_list = np.zeros((max_frames, 2))
        ball_velocity_list = np.zeros((max_frames, 2))
        pad_positions_list = np.zeros((max_frames, 4))
        pad_velocities_list = np.zeros((max_frames, 4))
    
        ball_position_list[0, :] = ball_position.reshape(1, 2)
        ball_velocity_list[0, :] = ball_velocity.reshape(1, 2)
        pad_positions_list[0, :] = pad_positions.reshape(1, 4)
        pad_velocities_list[0, :] = pad_velocities.reshape(1, 4)

    old_last_pad_id = None
    last_pad_id = None
    bounce_counter = np.asarray([0, 0, 0, 0])

    #print frames_shown, ball_position, ball_velocity, pad_positions, pad_velocities
    
    frames_shown = 1
    time = 0.0
    while not game_over and (not save_time_series or frames_shown < max_frames):

        # Update the pad accelerations by calling every agent
        for index, agent in enumerate(agents):
            agent.perceive_and_think(ball_position, ball_velocity, pad_positions, pad_velocities, court_width, court_height)
            # Simply set the pad_accelerations to desired_accelerations for now            
            pad_accelerations[index] = agent.desired_acceleration

        game_over, last_pad_id, time, ball_position, ball_velocity, pad_positions, pad_velocities, pad_accelerations = _time_evolution(time, ball_position, ball_velocity, last_pad_id, pad_positions, pad_velocities, pad_accelerations, pad_horizontal_offset, pad_height, court_width, court_height, delta_time)
        
        if save_time_series:
            ball_position_list[frames_shown, :] = ball_position.reshape(1, 2)
            ball_velocity_list[frames_shown, :] = ball_velocity.reshape(1, 2)
            pad_positions_list[frames_shown, :] = pad_positions.reshape(1, 4)
            pad_velocities_list[frames_shown, :] = pad_velocities.reshape(1, 4)

        if old_last_pad_id != last_pad_id and last_pad_id != None:
            bounce_counter[last_pad_id] += 1
            old_last_pad_id = last_pad_id
                

        frames_shown += 1        

    if save_time_series:
        # Discard unused data.
        ball_position_list = ball_position_list[:frames_shown - 1, :]
        ball_velocity_list = ball_velocity_list[:frames_shown - 1, :]
        pad_positions_list = pad_positions_list[:frames_shown - 1, :]
        pad_velocities_list = pad_velocities_list[:frames_shown - 1, :]    

        return last_pad_id, [ball_position_list, ball_velocity_list], [pad_positions_list, pad_velocities_list], bounce_counter
    else:
        ball_position = ball_position.reshape(1, 2)
        ball_velocity = ball_velocity.reshape(1, 2)
        pad_positions = pad_positions.reshape(1, 4)
        pad_velocities = pad_velocities.reshape(1, 4)

        return last_pad_id, ball_position, pad_positions + pad_height / 2.0, bounce_counter


