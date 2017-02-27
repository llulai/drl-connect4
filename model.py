from keras.models import Model
from keras.layers import Input, Convolution2D, Dense, Flatten, MaxPooling2D, merge
import keras.backend as K


def create_model():

    main_input = Input(shape=(6, 7, 2))
    cnn = Convolution2D(2, 3, 3, border_mode='same', activation='relu', init='normal')(main_input)
    cnn = MaxPooling2D(pool_size=(1,1), border_mode='same')(cnn)

    cnn = Convolution2D(4, 3, 3, border_mode='same', activation='relu', init='normal')(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2), border_mode='same')(cnn)

    cnn = Convolution2D(8, 3, 3, border_mode='same', activation='relu', init='normal')(cnn)
    cnn = MaxPooling2D(pool_size=(1, 1), border_mode='same')(cnn)

    cnn = Convolution2D(16, 3, 3, border_mode='same', activation='relu', init='normal')(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2), border_mode='same')(cnn)

    cnn = Convolution2D(32, 3, 3, border_mode='same', activation='relu', init='normal')(cnn)
    cnn = MaxPooling2D(pool_size=(1, 1), border_mode='same')(cnn)

    cnn = Flatten()(cnn)

    tile = Input(shape=(1,), name='tile')

    action = Input(shape=(7,), name='action')

    cnn = merge([cnn, action, tile], mode='concat')

    cnn = Dense(512, init='normal')(cnn)
    cnn = Dense(512, init='normal')(cnn)
    value = Dense(1, init='normal')(cnn)
    model = Model(input=[main_input, action, tile], output=value)
    model.compile(optimizer='rmsprop', loss='mse')

    return model
