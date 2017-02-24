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

    simulate(agents=agents, iterations=1000, log=True, print_every=100, backup=True)

if __name__ == '__main__':
    main()
