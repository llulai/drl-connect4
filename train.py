from agent import IntelligentAgent, LearningAgent
from simulation import simulate
import pickle
from keras.models import load_model
from model import create_model


def load_q():
    pickle_file = 'learning_agent.pickle'
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        q = save['learningagent']
        del save
    return q


def main():
    try:
        model = load_model('models/model_deep_q_learning.h5')
    except:
        model = create_model()

    #model = create_model(lr=0.25)

    la = LearningAgent(tile=1,
                       batch_size=10,
                       memory=50,
                       model=model,
                       alpha=0.5,
                       gamma=0.9,
                       exploration_rate=.05)

    la

    agents = [la, IntelligentAgent(-1, 1)]

    simulate(agents=agents, iterations=1000, log=True, print_every=100, backup=True)

if __name__ == '__main__':
    main()
    #from agent import parse_action, parse_tile, parse_state
    #from environment import get_initial_state
    #import numpy as np
    #state = get_initial_state()
    #state[0][0] = 1
    #parsed_state = parse_state(state)
    #
    #model = create_model()
    #p = model.predict(parsed_state)
    #q_sa = np.max(p)
    #print(p)
    #print(q_sa)

    #state = [[0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [1, 0, 0, 0, 0, 0, 0],
    #         [1, 0, 0, 0, 0, 0, 0],
    #         [1, 1, 1, 0, 0, 0, 0],
    #         ]
    #from agent import parse_state
    #print(parse_state(state,1))