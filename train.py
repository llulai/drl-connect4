from agent import IntelligentAgent, LearningAgent
from simulation import simulate
import pickle


def load_q():
    pickle_file = 'learning_agent.pickle'
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        q = save['learningagent']
        del save
    return q


agents = [LearningAgent(1), IntelligentAgent(-1, 1)]

agents = [LearningAgent(1), IntelligentAgent(-1, 1)]
#agents[0].Q = load_q()

simulate(agents=agents, iterations=100, log=True, train=True, print_every=10, backup=True)
