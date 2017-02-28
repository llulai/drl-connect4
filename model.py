from keras.models import Model
from keras.layers import Input, Convolution2D, Dense, Flatten, MaxPooling2D, Dropout
import keras.backend as K
from keras.optimizers import sgd


def create_model(lr=.2):

    main_input = Input(shape=(6, 7, 1))
    cnn = Convolution2D(2, 4, 4, border_mode='same', activation='relu', init='normal')(main_input)
    #cnn = MaxPooling2D(pool_size=(1,1), border_mode='same')(cnn)

    #cnn = Convolution2D(4, 3, 3, border_mode='same', activation='relu', init='normal')(cnn)
    #cnn = MaxPooling2D(pool_size=(1, 1), border_mode='same')(cnn)
    #cnn = Dropout(0.5)(cnn)

    #cnn = Convolution2D(8, 3, 3, border_mode='same', activation='relu', init='normal')(cnn)
    #cnn = MaxPooling2D(pool_size=(2, 2), border_mode='same')(cnn)

    #cnn = Convolution2D(16, 3, 3, border_mode='same', activation='relu', init='normal')(cnn)
    #cnn = MaxPooling2D(pool_size=(1, 1), border_mode='same')(cnn)
    #cnn = Dropout(0.5)(cnn)

    #cnn = Convolution2D(32, 3, 3, border_mode='same', activation='relu', init='normal')(cnn)
    #cnn = MaxPooling2D(pool_size=(1, 1), border_mode='same')(cnn)

    #cnn = Convolution2D(64, 3, 3, border_mode='same', activation='relu', init='normal')(cnn)
    #cnn = MaxPooling2D(pool_size=(2, 2), border_mode='same')(cnn)

    cnn = Flatten()(cnn)

    cnn = Dense(128, init='normal', activation='relu')(cnn)
    #cnn = Dense(128, init='normal', activation='relu')(cnn)
    #cnn = Dense(2048, init='normal', activation='relu')(cnn)
    #cnn = Dropout(0.5)(cnn)

    #cnn = Dense(1024, init='normal')(cnn)
    #cnn = Dropout(0.5)(cnn)

    value = Dense(7, init='normal', activation='tanh')(cnn)

    model = Model(input=main_input, output=value)
    model.compile(sgd(lr=lr), loss='mse')

    return model
