from agent import Agent, ActorCriticAgent, SearchAgent
from simulation import simulate


def main():
    # SETTINGS
    ac = ActorCriticAgent((1, -1))

    a = Agent((-1, 1))
    sa = SearchAgent(tiles=(1, -1), depth=1)

    simulate(agent=ac, sparring=a, opponent=a, iterations=10000, log=True, print_every=100, backup=True, learn_every=1)

if __name__ == '__main__':
    main()
