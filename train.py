from agent import IntelligentAgent, LearningAgent
from keras.models import load_model
from model import create_model
from simulation import simulate


def main():
    # SETTINGS

    try:
        model = load_model('models/model_1000.h5')
    except:
        model = create_model()

    model = create_model()

    la = LearningAgent(tile=1,
                       batch_size=1,
                       memory=1,
                       model=model,
                       alpha=0.5,
                       gamma=0.9,
                       exploration_rate=0)

    ia = IntelligentAgent(-1, 1)

    simulate(agent=la, opponent=ia, iterations=1000, start=1000, log=True, print_every=100, backup=False)

if __name__ == '__main__':
    main()
