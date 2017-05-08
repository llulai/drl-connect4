from agent import LearningAgent, SearchAgent
from model import create_model
from simulation import simulate

import tensorflow as tf
import pandas as pd


def main():
    # reset for variable names
    tf.reset_default_graph()

    # create the model for the agent
    model_la = create_model(lr=0.001, model_name='agent')

    la = LearningAgent(tiles=(1, -1),
                       batch_size=1,
                       memory=1,
                       model=model_la,
                       exploration_rate=.3)

    sa_0 = SearchAgent(tiles=(-1, 1), depth=0)
    sa_1 = SearchAgent(tiles=(-1, 1), depth=1)
    sa_2 = SearchAgent(tiles=(-1, 1), depth=2)

    sa = [sa_0, sa_1, sa_2]

    agent, results = simulate(agent=la, sparring=sa_1, opponents=[sa_0, sa_1], iterations=100, log=True, print_every=10, backup=True)
    df = pd.DataFrame(results)
    df.to_csv('results.csv')

if __name__ == '__main__':
    main()
