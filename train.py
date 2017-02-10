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


def main():
    agents = [LearningAgent(1, -1), LearningAgent(-1, 1)]
    #agents[0].Q = load_q()

    simulate(agents=agents, iterations=10000, log=True, print_every=100, backup=True)

if __name__ == '__main__':
    main()
