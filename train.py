from agent import IntelligentAgent, SuperAgent
from simulation import simulate


agents = [SuperAgent(tile=1, memory=100, batch_size=20), IntelligentAgent(-1, 1)]
simulate(agents=agents, iterations=1000, learn_after=25, log=True, train=True)
