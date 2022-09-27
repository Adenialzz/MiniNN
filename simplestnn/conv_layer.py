import numpy as np
from layer import Layer
from layer_utils import get_padding_2d, im2col, empty
from initializer import XavierUniformInit, ZerosInit, ConstantInit

class Conv2D(Layer):
    def __init__(self,
            kernel,
            stride=(1, 1),
            padding="SAME",
            w_init=XavierUniformInit(),
            b_init=ZerosInit()):
        super().__init__('Conv2D')
        self.kernel_shape = kernel
        self.stride = stride
        self.initializers = {'w': w_init, 'b': b_init}
        self.shapes = {'w': self.kernel_shape, 'b': self.kernel_shape[-1]}

        self.padding_mode = padding
        self.padding = None
        self.is_init = False

    def _init_parameters(self):
        self.params['w'] = self.initializers['w'](shape=self.shapes['w'])
        self.params['b'] = self.initializers['b'](shape=self.shapes['b'])
        self.is_init = True

    def _inputs_preprocess(self, inputs):
        _, in_h, in_w, _ = inputs.shape
        k_h, k_w, _, _ = self.kernel_shape
        # padding calculation
        if self.padding is None:
            self.padding = get_padding_2d(
                    (in_h, in_w), (k_h, k_w), self.padding_mode)
        return np.pad(inputs, pad_width=self.padding, mode='constant')

    def forward(self, inputs):
        if not self.is_init:
            self._init_parameters()

        k_h, k_w, _, out_c = self.kernel_shape
        s_h, s_w = self.stride

        X = self._inputs_preprocess(inputs)

        col = im2col(X, k_h, k_w, s_h, s_w)
        W = self.params['w'].reshape(-1, out_c)
        Z = col @ W
        batch_sz, in_h, in_w, _ = X.shape
        Z = Z.reshape(batch_sz, Z.shape[0] // batch_sz, out_c)
        out_h = (in_h - k_h) // s_h + 1
        out_w = (in_w - k_w) // s_w + 1
        Z = Z.reshape(batch_sz, out_h, out_w, out_c)

        Z += self.params['b']

        self.ctx = {'X_shape': X.shape, 'col': col, 'W': W}
        return Z

    def backward(self, grad):
        # read size
        k_h, k_w, in_c, out_c = self.kernel_shape
        s_h, s_w = self.stride
        batch_sz, in_h, in_w, in_c = self.ctx['X_shape']
        pad_h, pad_w = self.padding[1: 3]

        # grads wrt parameters
        flat_grad = grad.reshape(-1, out_c)
        d_W = self.ctx['col'].T @ flat_grad
        self.grads['w'] = d_W.reshape(self.kernel_shape)
        self.grads['b'] = np.sum(flat_grad, axis=0)

        # grads wrt inputs
        d_X = grad @ self.ctx['W'].T
        d_in = np.zeros(shape=self.ctx['X_shape'], dtype=np.float32)
        for i, r in enumerate(range(0, in_h - k_h + 1, s_h)):
            for j, c in enumerate(range(0, in_w - k_w + 1, s_w)):
                patch = d_X[:, i, j, :]
                patch = patch.reshape(batch_sz, k_h, k_w, in_c)
                d_in[:, r: r+k_h, c: c+k_w, :] += patch

        # cut off gradients of padding
        d_in = d_in[:, pad_h[0]: in_h - pad_h[1], pad_w[0]: in_w - pad_w[1], :]
        return d_in
            

class MaxPool2D(Layer):

    def __init__(self,
                 pool_size=(2, 2),
                 stride=None,
                 padding="VALID"):
        """Implement 2D max-pooling layer
        :param pool_size: A list/tuple of 2 integers (pool_height, pool_width)
        :param stride: A list/tuple of 2 integers (stride_height, stride_width)
        :param padding: A string ("SAME", "VALID")
        """
        super().__init__('MaxPool2D')
        self.kernel_shape = pool_size
        self.stride = stride if stride is not None else pool_size

        self.padding_mode = padding
        self.padding = None

    def forward(self, inputs):
        s_h, s_w = self.stride
        k_h, k_w = self.kernel_shape
        batch_sz, in_h, in_w, in_c = inputs.shape

        # zero-padding
        if self.padding is None:
            self.padding = get_padding_2d(
                (in_h, in_w), (k_h, k_w), self.padding_mode)
        X = np.pad(inputs, pad_width=self.padding, mode="constant")
        padded_h, padded_w = X.shape[1:3]

        out_h = (padded_h - k_h) // s_h + 1
        out_w = (padded_w - k_w) // s_w + 1

        # construct output matrix and argmax matrix
        max_pool = empty((batch_sz, out_h, out_w, in_c))
        argmax = empty((batch_sz, out_h, out_w, in_c), dtype=int)
        for r in range(out_h):
            r_start = r * s_h
            for c in range(out_w):
                c_start = c * s_w
                pool = X[:, r_start: r_start+k_h, c_start: c_start+k_w, :]
                pool = pool.reshape((batch_sz, -1, in_c))

                _argmax = np.argmax(pool, axis=1)[:, np.newaxis, :]
                argmax[:, r, c, :] = _argmax.squeeze()

                # get max elements
                _max_pool = np.take_along_axis(pool, _argmax, axis=1).squeeze()
                max_pool[:, r, c, :] = _max_pool

        self.ctx = {"X_shape": X.shape, "out_shape": (out_h, out_w),
                    "argmax": argmax}
        return max_pool

    def backward(self, grad):
        batch_sz, in_h, in_w, in_c = self.ctx["X_shape"]
        out_h, out_w = self.ctx["out_shape"]
        s_h, s_w = self.stride
        k_h, k_w = self.kernel_shape
        k_sz = k_h * k_w
        pad_h, pad_w = self.padding[1:3]

        d_in = np.zeros(shape=(batch_sz, in_h, in_w, in_c), dtype=np.float32)
        for r in range(out_h):
            r_start = r * s_h
            for c in range(out_w):
                c_start = c * s_w
                _argmax = self.ctx["argmax"][:, r, c, :]
                mask = np.eye(k_sz)[_argmax].transpose((0, 2, 1))
                _grad = grad[:, r, c, :][:, np.newaxis, :]
                patch = np.repeat(_grad, k_sz, axis=1) * mask
                patch = patch.reshape((batch_sz, k_h, k_w, in_c))
                d_in[:, r_start:r_start+k_h, c_start:c_start+k_w, :] += patch

        # cut off gradients of padding
        return d_in[:, pad_h[0]: in_h-pad_h[1], pad_w[0]: in_w-pad_w[1], :]
class Reshape(Layer):

    def __init__(self, *output_shape):
        super().__init__('Reshape')
        self.output_shape = output_shape
        self.input_shape = None

    def forward(self, inputs):
        self.input_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], *self.output_shape)

    def backward(self, grad):
        return grad.reshape(self.input_shape)


class Flatten(Reshape):

    def __init__(self):
        super().__init__(-1)


