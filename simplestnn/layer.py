import numpy as np
from initializer import XavierUniformInit, ZerosInit

class Layer:
    def __init__(self, name):
        self.name = name
        self.params = {}
        self.grads = {}
        self.is_training = True

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError
    
    def set_phase(self, phase):
        self.is_training = True if phase == 'TRAIN' else False

class Dense(Layer):
    def __init__(self, num_in, num_out, w_init=XavierUniformInit(), b_init=ZerosInit()):
        super().__init__('Linear')
        self.shapes = {'w': [num_in, num_out], 'b': [1, num_out]}
        self.params = { 'w': None, 'b': None }
        self.initializers = {'w': w_init, 'b': b_init}

        self.is_init = False
        self.inputs = None
        if num_in is not None:
            self._init_parameters(num_in)

        self.inputs = None

    def forward(self, inputs):
        # lazy init
        if not self.is_init:
            self._init_parameters(inputs.shape[1])

        self.inputs = inputs
        return inputs @ self.params['w'] + self.params['b']

    def backward(self, grad):
        self.grads['w'] = self.inputs.T @ grad
        self.grads['b'] = np.sum(grad, axis=0)
        return grad @ self.params['w'].T
    
    def _init_parameters(self, input_size):
        self.shapes['w'][0] = input_size
        self.params['w'] = self.initializers['w'](shape=self.shapes['w'])
        self.params['b'] = self.initializers['b'](shape=self.shapes['b'])
        self.is_init = True

class Activation(Layer):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return self.func(inputs)

    def backward(self, grad):
        return self.dericative_func(self.inputs) * grad

    def func(self, x):
        raise NotImplementedError

    def dericative_func(self, x):
        raise NotImplementedError

class ReLU(Activation):
    def __init__(self):
        super().__init__('ReLU')

    def func(self, x):
        return np.maximum(x, 0)

    def dericative_func(self, x):
        return x > 0.0

