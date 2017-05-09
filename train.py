from agent import LearningAgent, SearchAgent
from model import create_model
from simulation import simulate

import tensorflow as tf
import pickle


def main(model_name, memory, batch_size, exploration_rate, gamma, test_every, iterations):
    # reset for variable names
    tf.reset_default_graph()

    # create the model for the agent
    model_la = create_model(lr=0.001, model_name=model_name)

    learning_agent = LearningAgent(tiles=(1, -1),
                                   memory=memory,
                                   batch_size=batch_size,
                                   exploration_rate=exploration_rate,
                                   gamma=gamma,
                                   model=model_la)

    sa_0 = SearchAgent(tiles=(-1, 1), depth=0)
    sa_1 = SearchAgent(tiles=(-1, 1), depth=1)

    opponents = [sa_0, sa_1]
    results = []

    for i, opponent in enumerate(opponents, 1):
        print('training {} of {} opponents'.format(i, len(opponents)))

        results += simulate(agent=learning_agent,
                            sparring=opponent,
                            opponents=opponents,
                            iterations=iterations,
                            test_every=test_every,
                            backup=False)

    with open('logs/results.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results


if __name__ == '__main__':
    model_name = 'deep_neural_network'
    memory = 1
    batch_size = 1
    exploration_rate = .3
    gamma = .9
    test_every = 100
    iterations = 1000

    main(model_name, memory, batch_size, exploration_rate, gamma, test_every, iterations)
