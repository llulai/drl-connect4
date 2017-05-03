from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop


def create_model(lr=.001):

    main_input = Input(shape=(42,))
    cnn = Dense(32, init='normal', activation='relu')(main_input)
    cnn = Dense(64, init='normal', activation='relu')(cnn)
    cnn = Dense(128, init='normal', activation='relu')(cnn)

    value = Dense(512, init='normal', activation='relu')(cnn)
    value = Dense(512, init='normal', activation='relu')(value)
    value = Dense(1, init='normal', activation='tanh')(value)

    model = Model(input=main_input, output=value)

    model.compile(optimizer=RMSprop(lr=lr), loss='mse')

    return model
