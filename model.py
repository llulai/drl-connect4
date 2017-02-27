from keras.models import Model
from keras.layers import Input, Convolution2D, Dense, Flatten, MaxPooling2D, Dropout
import keras.backend as K


def create_model():

    main_input = Input(shape=(6, 7, 1))
    cnn = Convolution2D(2, 3, 3, border_mode='same', activation='relu', init='normal')(main_input)
    cnn = MaxPooling2D(pool_size=(1,1), border_mode='same')(cnn)

    cnn = Convolution2D(4, 3, 3, border_mode='same', activation='relu', init='normal')(cnn)
    cnn = MaxPooling2D(pool_size=(1, 1), border_mode='same')(cnn)
    cnn = Dropout(0.5)(cnn)

    cnn = Convolution2D(8, 3, 3, border_mode='same', activation='relu', init='normal')(cnn)
    cnn = MaxPooling2D(pool_size=(1, 1), border_mode='same')(cnn)

    cnn = Convolution2D(16, 3, 3, border_mode='same', activation='relu', init='normal')(cnn)
    cnn = MaxPooling2D(pool_size=(1, 1), border_mode='same')(cnn)
    nn = Dropout(0.5)(cnn)

    cnn = Convolution2D(32, 3, 3, border_mode='same', activation='relu', init='normal')(cnn)
    cnn = MaxPooling2D(pool_size=(1, 1), border_mode='same')(cnn)

    cnn = Convolution2D(64, 3, 3, border_mode='same', activation='relu', init='normal')(cnn)
    cnn = MaxPooling2D(pool_size=(1, 1), border_mode='same')(cnn)
    cnn = Dropout(0.5)(cnn)

    cnn = Flatten()(cnn)

    cnn = Dense(2048, init='normal')(cnn)
    cnn = Dropout(0.5)(cnn)

    cnn = Dense(1024, init='normal')(cnn)
    cnn = Dropout(0.5)(cnn)

    value = Dense(1, init='normal')(cnn)

    model = Model(input=main_input, output=value)
    model.compile(optimizer='rmsprop', loss='mse')

    return model
