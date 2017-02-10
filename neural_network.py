import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                       (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5,
                                       (self.output_nodes, self.hidden_nodes))

        # Initialize traces
        self.e_input_to_hidden = np.zeros((self.hidden_nodes, self.input_nodes))
        self.e_hidden_to_output = np.zeros((self.output_nodes, self.hidden_nodes))

        self.lr = learning_rate

        #### Set this to your implemented sigmoid function ####
        # Activation function is the sigmoid function
        self.activation_function = lambda x: 1./(1+np.exp(-x))
        self.vectorized_activation_function = np.vectorize(self.activation_function)

    def predict(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T

        # hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)  # signals into hidden layer
        hidden_outputs = self.vectorized_activation_function(hidden_inputs)  # signals from hidden layer

        # output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)  # signals into final output layer
        final_outputs = self.vectorized_activation_function(final_inputs)  # signals from final output layer

        return final_outputs

    def train(self, st, st_1, reward):
        # Convert inputs list to 2d array
        inputs = np.array(st, ndmin=2).T

        # hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)  # signals into hidden layer
        hidden_outputs = self.vectorized_activation_function(hidden_inputs)  # signals from hidden layer

        # output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)  # signals into final output layer
        final_outputs = final_inputs  # signals from final output layer

        output_grad = final_outputs * (1 - final_outputs) * hidden_outputs.T
        self.e_hidden_to_output += output_grad

        hidden_grad = np.dot(hidden_outputs * (1 - hidden_outputs) * output_grad.T, inputs.T)
        self.e_input_to_hidden += hidden_grad

        p_st1 = self.predict(st_1)

        self.weights_hidden_to_output += self.lr * (reward + p_st1 - final_outputs) * self.e_hidden_to_output
        self.weights_input_to_hidden += self.lr * (reward + p_st1 - final_outputs) * self.e_input_to_hidden


def parse_game(game):
    X = []
    rewards = []
    for turn in game:
        x = list(np.array(turn).reshape((-1)))
        reward = get_winner(turn)

        X.append(x)
        rewards.append(reward)

    return X, rewards

if __name__ == '__main__':
    from agent import Agent
    from simulation import simulate
    from environment import get_winner

    agents = [Agent(1), Agent(-1)]
    _, results = simulate(agents=agents, iterations=1, log=False, print_every=1, backup=False)

    X, rewards = parse_game(results[0])

    nn = NeuralNetwork(42, 5, 1, 0.1)
    nn.train(X[0], X[1], rewards[0])

    #print('%s prediction' %prediction)

    #print(rewards)

