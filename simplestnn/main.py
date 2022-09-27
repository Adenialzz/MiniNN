from layer import Dense, ReLU
from conv_layer import Conv2D, MaxPool2D, Flatten

from net import Net
from loss import SoftmaxCrossEntropyLoss
from optimizer import Adam
from model import Model
from utils import MNIST, BatchIterator

import numpy as np
import time

useCnn = True

if useCnn:
    # A LeNet-5 model with activation function changed to ReLU
    net = Net([
        Conv2D(kernel=[5, 5, 1, 6], stride=[1, 1]),
        ReLU(),
        MaxPool2D(pool_size=[2, 2], stride=[2, 2]),
        Conv2D(kernel=[5, 5, 6, 16], stride=[1, 1]),
        ReLU(),
        MaxPool2D(pool_size=[2, 2], stride=[2, 2]),
        Flatten(),
        Dense(28*28, 120),
        ReLU(),
        Dense(120, 84),
        ReLU(),
        Dense(84, 10)
    ])
else:
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
if useCnn:
    train_x = train_x.reshape((-1, 28, 28, 1))
    test_x = test_x.reshape((-1, 28, 28, 1))

iterator = BatchIterator(128, True)

for epoch in range(10):
    t0 = time.perf_counter()
    for batch in iterator(train_x, train_y):
        pred = model.forward(batch.inputs)
        loss, grads = model.backward(pred, batch.targets)
        model.apply_grad(grads)
    test_pred = model.forward(test_x)
    test_pred_idx = np.argmax(test_pred, axis=1)
    test_y_idx = np.argmax(test_y, axis=1)
    correct_num = int(np.sum(test_pred_idx == test_y_idx))
    t1 = time.perf_counter()
    print(f"total: {len(test_pred)}, correct: {correct_num}, accuracy: {(correct_num / len(test_pred)):.4f}, duration: {(t1 - t0):.2f}s")






