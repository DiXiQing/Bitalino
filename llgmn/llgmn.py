"""
Edited from https://github.com/kwsk-h/LLGMN/
"""

import numpy as np


class LLGMN:
    def __init__(self):
        self.D = 4
        self.H = int(1 + self.D * (self.D + 3) / 2)
        self.K = 4
        self.M = 3
        self.epsilon = 0.1
        self.batch_size = 1
        self.max_epoch = 500
        self.data = []
        self.weight = np.random.rand(self.K, self.M, self.H)
        self.IO = {}

    def load_weight(self, path):
        weight = np.load(path)
        assert self.weight.shape == weight.shape, "Weights have incorrect shape!"
        self.weight = weight

    def save_weight(self, path):
        np.save(path, self.weight)

    def inputConversion(self, input_data):
        conv_data = [1]
        conv_data.extend(input_data)
        tmp = [input_data[i] * input_data[j] for i in range(self.D) for j in range(i, self.D)]
        conv_data.extend(tmp)
        return np.array(conv_data)

    def forward(self, batch_data):
        I1 = np.array([self.inputConversion(x) for x in batch_data])
        O1 = I1
        I2 = np.array([np.sum(o1 * self.weight, axis=2) for o1 in O1])
        O2 = np.array([np.exp(i2) / np.sum(np.exp(i2)) for i2 in I2])
        I3 = np.sum(O2, axis=2)
        O3 = I3

        self.IO = {"I1": I1, "O1": O1, "I2": I2, "O2": O2, "I3": I3, "O3": O3}

        return self.smooth_output(O3)

    def smooth_output(self, output, alpha=0.1):
        smoothed = np.zeros_like(output)
        smoothed[0] = output[0]
        for t in range(1, len(output)):
            smoothed[t] = alpha * output[t] + (1 - alpha) * smoothed[t-1]
        return smoothed

    def backward(self, batch_T):
        grad = [
            ((self.IO["O3"][i] - batch_T[i]).reshape(self.K, 1) * (self.IO["O2"][i] / self.IO["O3"][i].reshape(self.K, 1))).reshape(self.K, self.M, 1) * self.IO["I1"][i]
            for i in range(self.batch_size)
        ]
        self.grad = np.sum(grad, axis=0) / self.batch_size

    def train(self, data, label):
        for epoch in range(self.max_epoch):
            lr = self.epsilon / (1 + epoch / 100)
            iter_per_epoch = max(int(len(data) / self.batch_size), 1)
            for i in range(iter_per_epoch):
                idx = np.random.choice(len(data), size=self.batch_size, replace=False)
                _ = self.forward(data[idx])
                self.backward(label[idx])
                self.weight -= lr * self.grad
            Y = self.forward(data)
            loss = np.sum((Y - label) ** 2)
            entropy = np.sum(-1 * label * np.log(Y)) / len(data)
            acc = sum(np.argmax(t) == np.argmax(y) for t, y in zip(label, Y)) / len(data)
            print(epoch, ":", acc, loss, entropy)
        return Y

    def test(self, data, label):
        Y = self.forward(data)
        acc = sum(np.argmax(t) == np.argmax(y) for t, y in zip(label, Y)) / len(data)
        print("Accurcy : ", acc)
        return Y



if __name__ == "__main__":
    data_train = np.loadtxt("llgmn/DXQ_data/test.csv", delimiter=",")
    label_train = np.loadtxt("llgmn/DXQ_data/label.csv", delimiter=",")
    ##data_train = np.loadtxt("llgmn/data/data_train_movement_24_655.csv", delimiter=",")
    ##label_train = np.loadtxt("llgmn/data/twenty_january_label_train_movement.csv", delimiter=",")
    ll = LLGMN()
    Y_train = ll.train(data_train, label_train)

    ll.save_weight("llgmn/data/dxq_movement_weights.npy")
