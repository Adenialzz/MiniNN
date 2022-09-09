from layer import Dense, ReLU
from net import Net
from loss import SoftmaxCrossEntropyLoss
from optimizer import Adam
from model import Model
from utils import MNIST, BatchIterator

import numpy as np

net = Net([
    Dense(28*28, 400),
    ReLU(),
    Dense(400, 100),
    ReLU(),
    Dense(100, 10)
])

optimizer = Adam(lr=1e-3)
model = Model(net, SoftmaxCrossEntropyLoss(), optimizer)

mnist = MNIST('./data', one_hot=True)

train_x, train_y = mnist.train_set
test_x, test_y = mnist.test_set

iterator = BatchIterator(128, True)
for epoch in range(10):
    for batch in iterator(train_x, train_y):
        pred = model.forward(batch.inputs)
        loss, grads = model.backward(pred, batch.targets)
        model.apply_grad(grads)
    test_pred = model.forward(test_x)
    test_pred_idx = np.argmax(test_pred, axis=1)
    test_y_idx = np.argmax(test_y, axis=1)
    correct_num = int(np.sum(test_pred_idx == test_y_idx))
    print(f"total: {len(test_pred)}, correct: {correct_num}, accuracy: {correct_num / len(test_pred)}")






