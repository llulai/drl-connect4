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
        model = load_model('1model.h5')
    except:
        model = create_model()

    model = create_model()

    agents = [LearningAgent(tile=1, batch_size=10, memory=50, model=model), IntelligentAgent(-1, 1)]

    simulate(agents=agents, iterations=1000, log=True, print_every=10, backup=False)

if __name__ == '__main__':
    main()
    #from agent import parse_action, parse_tile, parse_state
    #from environment import get_initial_state
    #state = parse_state(get_initial_state())
    #action = parse_action(1)
    #tile = parse_tile(1)
    #print(action.shape)


    #model = create_model()
    #p = model.predict([state, action, tile])
    #print(p)
    #state = [[0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [1, 0, 0, 0, 0, 0, 0],
    #         [1, 0, 0, 0, 0, 0, 0],
    #         [1, 1, 1, 0, 0, 0, 0],
    #         ]
    #from agent import parse_state
    #print(parse_state(state,1))