from agent import Agent, IntelligentAgent, LearningAgent, SearchAgent
from model import create_model
from simulation import simulate

import tensorflow as tf


def main():
    # reset for variable names
    tf.reset_default_graph()

    # create the model for the agent
    model_la = create_model(lr=0.001, model_name='agent')

    #model_sp = create_model(lr=0.001)

    la = LearningAgent(tiles=(1, -1),
                       batch_size=1,
                       memory=1,
                       model=model_la,
                       gamma=0.9,
                       exploration_rate=.3)

    #sp = LearningAgent(tiles=(-1, 1),
    #                   batch_size=1,
    #                   memory=1,
    #                   model=model_sp,
    #                   gamma=0.9,
    #                   exploration_rate=.3)

    ia = IntelligentAgent((-1, 1))

    a = Agent((-1, 1))
    sa = SearchAgent(tiles=(-1, 1), depth=2)

    simulate(agent=la, sparring=sa, opponent=a, iterations=5000, log=True, print_every=10, backup=True)

if __name__ == '__main__':
    main()
