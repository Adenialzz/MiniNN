
class Model:
    def __init__(self, net, loss, optimizer):
        self.net = net
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, inputs):
        return self.net.forward(inputs)

    def backward(self, preds, labels):
        # 计算损失和梯度
        loss = self.loss.loss(preds, labels)
        grad = self.loss.grad(preds, labels)
        grads = self.net.backward(grad)
        return loss, grads

    def apply_grad(self, grads):
        # 根据计算得到的梯度更新参数
        params = self.net.get_params()
        steps = self.optimizer.compute_step(grads, params)
        for step, param in zip(steps, params):
            for k in param.keys():
                param[k] += step[k]

