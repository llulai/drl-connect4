from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras import backend as k
import numpy as np


def create_model():
    main_input = Input(shape=42)
    layer = Dense(512, init='normal', activation='sigmoid')(main_input)
    output = Dense(7, init='normal', activation='softmax')(layer)
    model = Model(input=main_input, output=output)

    return main_input, output, model


def create_actor():
    main_input = Input(shape=(42,))
    layer = Dense(128, activation='relu')(main_input)
    layer = Dense(64, activation='relu')(layer)
    layer = Dense(32, activation='relu')(layer)
    output = Dense(7, activation='softmax')(layer)
    model = Model(input=main_input, output=output)

    return main_input, output, model


def create_critic():
    main_input = Input(shape=(42,))
    layer = Dense(128, activation='relu')(main_input)
    layer = Dense(64, activation='relu')(layer)
    layer = Dense(32, activation='relu')(layer)
    output = Dense(1)(layer)
    model = Model(input=main_input, output=output)

    return main_input, output, model


class BaseModel(object):
    def __init__(self, session, model_func=None):
        self.input, output, self.model = model_func or create_model()
        self.grad_func = k.gradients(output, self.model.weights)
        self.sess = session

    def get_gradient(self, parsed_state):
        return np.array(self.sess.run(self.grad_func, feed_dict={self.input: parsed_state}))

    def predict(self, parsed_state):
        return self.model.predict(parsed_state)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, w):
        self.model.set_weights(w)

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = load_model(filename)


class ActorModel(BaseModel):
    def __init__(self, session):
        BaseModel.__init__(self, session=session, model_func=create_actor())

    #def get_gradient(self, parsed_state, action):
    #    parsed_action = np.zeros((1, 7))
    #    parsed_action[0][action] = 1#
    #
    #    return self.sess.run(self.grad_func,
    #                         feed_dict={
    #                             self.input: parsed_state,
    #                             self.model.output: parsed_action
    #                         })


class CriticModel(BaseModel):
    def __init__(self, session):
        BaseModel.__init__(self, session=session, model_func=create_critic())

    #def get_gradient(self, parsed_state):
    #    return self.sess.run(self.grad_func, feed_dict={self.input: parsed_state})
