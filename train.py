from agent import Agent, IntelligentAgent, ActorCriticAgent
from simulation import simulate


def main():
    # SETTINGS
    ac = ActorCriticAgent((1, -1))

    ia = IntelligentAgent((-1, 1))

    a = Agent((-1, 1))
    # sa = SearchAgent(tiles=(1, -1), depth=2)

    simulate(agent=ac, sparring=a, opponent=a, iterations=10000, log=True, print_every=1, backup=True, learn_every=1)

if __name__ == '__main__':
    main()
