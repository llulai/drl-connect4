from agent import LearningAgent, SearchAgent
from model import create_model
from simulation import simulate

import tensorflow as tf


def main(model_name, memory, batch_size, exploration_rate, gamma):
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

    for opponent in opponents:

        results += simulate(agent=learning_agent,
                            sparring=opponent,
                            opponents=opponents,
                            iterations=10,
                            log=True,
                            print_every=1,
                            backup=False)

    return results

    # results must have
    # iterations
    # model against is trained
    # model against is playing
    # game
    # reward

if __name__ == '__main__':
    model_name = 'conv_neural_network'
    memory = 1
    batch_size = 1
    exploration_rate = .3
    gamma = .9

    main(model_name, memory, batch_size, exploration_rate, gamma)
