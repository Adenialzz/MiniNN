import numpy as np

class BaseOptimizer:
    def __init__(self, lr, weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay

    def compute_step(self, grads, params):
        steps = list()
        # flatten all gradients
        flatten_grads = np.concatenate(
            [np.ravel(v) for grad in grads for v in grad.values()])
        # compute step
        flatten_step = self._compute_step(flatten_grads)
        # reshape gradients
        p = 0
        for param in params:
            layer = dict()
            for k, v in param.items():
                block = np.prod(v.shape)
                _step = flatten_step[p: p+block].reshape(v.shape)
                _step -= self.weight_decay * v
                layer[k] = _step
                p += block
            steps.append(layer)
        return steps

    def _compute_step(self, grad):
        raise NotImplementedError

class Adam(BaseOptimizer):
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._b1 = beta1
        self._b2 = beta2
        self._eps = eps

        self._t = 0
        self._m = 0
        self._v = 0

    def _compute_step(self, grad):
        self._t += 1

        self._m += (1.0 - self._b1) * (grad - self._m)
        self._v += (1.0 - self._b2) * (grad ** 2 - self._v)

        _m = self._m / (1 - self._b1 ** self._t)
        _v = self._v / (1 - self._b2 ** self._t)

        step = -self.lr * _m / (_v ** 0.5 + self._eps)
        return step

