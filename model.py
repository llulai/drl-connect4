from keras.models import Model
from keras.layers import Input, Convolution2D, Dense, Flatten, MaxPooling2D
import keras.backend as K

def create_model():

    main_input = Input(shape=(6, 7, 1))
    cnn = Convolution2D(2, 3, 3, border_mode='same')(main_input)
    cnn = MaxPooling2D(pool_size=(1,1), border_mode='same')(cnn)

    cnn = Convolution2D(4, 3, 3, border_mode='same')(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2), border_mode='same')(cnn)

    cnn = Convolution2D(8, 3, 3, border_mode='same')(cnn)
    cnn = MaxPooling2D(pool_size=(1, 1), border_mode='same')(cnn)

    cnn = Convolution2D(16, 3, 3, border_mode='same')(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2), border_mode='same')(cnn)

    cnn = Convolution2D(32, 3, 3, border_mode='same')(cnn)
    cnn = MaxPooling2D(pool_size=(1, 1), border_mode='same')(cnn)

    cnn = Flatten()(cnn)
    cnn = Dense(512)(cnn)
    action = Dense(7, activation='softmax')(cnn)
    model = Model(input=main_input, output=action)
    model.compile(optimizer='rmsprop', loss='mse')

    return model