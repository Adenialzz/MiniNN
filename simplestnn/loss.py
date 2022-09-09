import numpy as np

# ===================== math utils ===================== #

def softmax(x, t=1.0, axis=-1):
    x_ = x / t
    x_max = np.max(x_, axis=axis, keepdims=True)
    exps = np.exp(x_ - x_max)
    return exps / np.sum(exps, axis=axis, keepdims=True)

def log_softmax(x, t=1.0, axis=-1):
    x_ = x / t
    x_max = np.max(x_, axis=axis, keepdims=True)
    exps = np.exp(x_ - x_max)
    exp_sum = np.sum(exps, axis=axis, keepdims=True)
    return x_ - x_max - np.log(exp_sum)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class BaseLoss:
    def loss(self, pred, label):
        raise NotImplementedError

    def grad(self, pred, label):
        raise NotImplementedError

# ===================== loss ===================== #

class SoftmaxCrossEntropyLoss(BaseLoss):

    def __init__(self, weight=None):
        """
        L = weight[class] * (-log(exp(x[class]) / sum(exp(x))))
        :param weight: A 1D tensor [n_classes] assigning weight to each corresponding sample.
        """
        weight = np.asarray(weight) if weight is not None else weight
        self._weight = weight

    def loss(self, logits, labels):
        m = logits.shape[0]
        exps = np.exp(logits - np.max(logits))
        p = exps / np.sum(exps)
        nll = -np.log(np.sum(p * labels, axis=1))

        if self._weight is not None:
            nll *= self._weight[labels]
        return np.sum(nll) / m

    def grad(self, logits, labels):
        m = logits.shape[0]
        grad = np.copy(logits)
        grad -= labels
        return grad / m
