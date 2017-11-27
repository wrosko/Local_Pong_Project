from abc import ABCMeta, abstractmethod
import numpy as np
# import tensorflow as tf

class Agent(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._desired_acceleration = 0.0
    
    @abstractmethod
    def perceive_and_think(self, ball_position, ball_velocity, pad_positions, pad_velocities, court_width, court_height):
        raise NotImplementedError

    @property
    def desired_acceleration(self):
        return self._desired_acceleration

    @desired_acceleration.setter
    def desired_acceleration(self, value):
        self._desired_acceleration = value
    
class RandomAgent(Agent):
    
    def __init__(self, activity_level):
        super(RandomAgent, self).__init__()
        self.__activity_level = activity_level

    def perceive_and_think(self, ball_position, ball_velocity, pad_positions, pad_velocities, court_width, court_height):        
        # Possibly return a random desired_acceleration
        if np.random.random() <= self.__activity_level:
            self.desired_acceleration = 2 * np.random.random() - 1
        else:
            self.desired_acceleration = 0.0

            
class TwoLayerAgent(Agent):

    def __init__(self, hidden_weights, hidden_biases, output_weights, output_bias):
        super(TwoLayerAgent, self).__init__()

        # Given the numpy weights, convert them into tensorflow tensor.
        self.__n_network_inputs = 2 + 2 + 4 + 4 + 1 + 1 # ball pos, ball vel, pad poss, pad vels, width, height (14)


        self.__hidden_weights = hidden_weights
        self.__hidden_biases = hidden_biases
        self.__output_weights = output_weights
        self.__output_bias = output_bias

        
    def perceive_and_think(self, ball_position, ball_velocity, pad_positions, pad_velocities, court_width, court_height):
        assert ball_position.shape == (2, 1)
        assert ball_velocity.shape == (2, 1)
        assert pad_positions.shape == (4, 1)
        assert pad_velocities.shape == (4, 1)

        court_width = np.atleast_2d(court_width)
        assert court_width.shape == (1, 1)
        
        court_height = np.atleast_2d(court_height)
        assert court_height.shape == (1, 1)
        
        n_inputs = self.__n_network_inputs

        # Store the inputs in a numpy array.
        concatenatees = (ball_position.T,
                         ball_velocity.T,
                         pad_positions.T,
                         pad_velocities.T,
                         court_width,
                         court_height)
                                 
        inputs = np.concatenate(concatenatees, axis=1).T#astype('float32')

        hidden_weights = self.__hidden_weights
        hidden_biases = self.__hidden_biases
        output_weights = self.__output_weights
        output_bias = self.__output_bias        

        local_field_hidden = np.dot(hidden_weights, inputs) + hidden_biases
        neuron_activation_levels = np.tanh(local_field_hidden)

        local_field_output = np.dot(output_weights, neuron_activation_levels) + output_bias
        acceleration = np.tanh(local_field_output)
        
        self.desired_acceleration = float(acceleration)


# class TensorFlowAgent(Agent):
#
#     def __init__(self, hidden_weights, hidden_biases, output_weights, output_bias):
#         super(TwoLayerAgent, self).__init__()
#
#         # Given the numpy weights, convert them into tensorflow tensor.
#         self.__n_network_inputs = 2 + 2 + 4 + 4 + 1 + 1 # ball pos, ball vel, pad poss, pad vels, width, height (14)
#
#         self.__tf_session = tf.InteractiveSession() # Setup a tensorflow session
#         self.__initialize_network(hidden_weights, hidden_biases, output_weights, output_bias)
#
#
#     def __initialize_network(self, hidden_weights, hidden_biases, output_weights, output_bias):
#         n_inputs = self.__n_network_inputs
#
#         assert n_inputs == hidden_weights.shape[1] # Assert that number of columns is equal to the number of inputs
#
#         n_hidden_neurons = hidden_weights.shape[0]
#         assert n_hidden_neurons == len(hidden_biases)
#
#         assert output_weights.shape[0] == 1 # Assert that we output a scalar (acceleration). For that, we must multiply by a row.
#         assert n_hidden_neurons == output_weights.shape[1] # The number of hidden neurons is equal to the number of inputs to the output layer.
#
#         # Tensorflow has its own representation of variables. We need to convert the numpy stuff to their format. Setup a helper function
#         def convert_to_tensorflow_variable(name, matrix):
#             matrix = np.asarray(matrix) # Ensure it's a numpy matrix (because the scalar bias)
#             #initializer = tf.constant_initializer(matrix)
#             #variable = tf.get_variable(name, initializer = initializer, shape = matrix.shape)
#
#             constant = tf.constant(matrix, tf.float32)
#             variable = tf.Variable(tf.zeros(matrix.shape)).assign(constant)
#             return variable
#
#         # Do the conversions
#         hidden_weights = convert_to_tensorflow_variable('hidden_weights', hidden_weights)
#         hidden_biases = convert_to_tensorflow_variable('hidden_biases', hidden_biases)
#         output_weights = convert_to_tensorflow_variable('output_weights', output_weights)
#         output_bias = convert_to_tensorflow_variable('output_bias', output_bias)
#
#         # Now, lets set up the model, a.k.a a a string of operations that tensorflow calls a graph.
#         # Tensorflow graphs are created by nesting tensorflow variables and functions to eachother as follows:
#
#         inputs = tf.placeholder(tf.float32, [n_inputs, 1]) # Create placeholder for input vector to be passed to tensorflow.
#         self.__inputs = inputs # Save it for later
#
#         local_field_hidden = tf.matmul(hidden_weights, inputs) + hidden_biases
#         neuron_activation_levels = tf.nn.tanh(local_field_hidden)  # TODO: chosen arbitrarily, we should probably think about this function
#         local_field_output = tf.matmul(output_weights, neuron_activation_levels) + output_bias
#         output = tf.nn.tanh(local_field_output)
#
#         tf.global_variables_initializer().run() # Do the actual initialization of the variables we declared above.
#
#         # The tensorflow graph is stored in 'output'. Let's save it for later use.
#         self.__network_graph = output
#
#
#     def perceive_and_think(self, ball_position, ball_velocity, pad_positions, pad_velocities, court_width, court_height):
#         assert ball_position.shape == (2, 1)
#         assert ball_velocity.shape == (2, 1)
#         assert pad_positions.shape == (4, 1)
#         assert pad_velocities.shape == (4, 1)
#
#         court_width = np.atleast_2d(court_width)
#         assert court_width.shape == (1, 1)
#
#         court_height = np.atleast_2d(court_height)
#         assert court_height.shape == (1, 1)
#
#         n_inputs = self.__n_network_inputs
#
#         # Store the inputs in a numpy array.
#         concatenatees = (ball_position.T,
#                          ball_velocity.T,
#                          pad_positions.T,
#                          pad_velocities.T,
#                          court_width,
#                          court_height)
#
#         input_values = np.concatenate(concatenatees, axis=1).T#astype('float32')
#
#         output = self.__network_graph
#         session = self.__tf_session
#         inputs = self.__inputs # Get the placeholder
#
#         acceleration = session.run(output, feed_dict = {inputs: input_values})
#         self.desired_acceleration = float(acceleration)
#
#

        
                        

                         
                                                                                                              
        

        
