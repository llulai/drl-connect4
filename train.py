from agent import Agent, IntelligentAgent, LearningAgent
from keras.models import load_model
from model import create_model
from simulation import simulate


def main():
    # SETTINGS

    try:
        model = load_model('models/agent.h5')
    except:
        model = create_model(lr=0.001)

    model.optimizer.lr.assign(0.00001)

    la = LearningAgent(tiles=(1, -1),
                       batch_size=1,
                       memory=1,
                       model=model,
                       gamma=0.9,
                       exploration_rate=.3)

    sp = LearningAgent(tiles=(-1, 1),
                       batch_size=1,
                       memory=1,
                       model=model,
                       gamma=0.9,
                       exploration_rate=.3)

    ia = IntelligentAgent((-1, 1))

    a = Agent((-1, 1))

    simulate(agent=la, sparring=sp, opponent=ia, iterations=10000, log=True, print_every=1000, backup=True)

if __name__ == '__main__':
    main()
