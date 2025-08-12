"""
Edited from https://github.com/kwsk-h/LLGMN/
"""

import numpy as np


class LLGMN:
    def __init__(self):
        self.D = 4  # input dimension
        self.H = int(1 + self.D * (self.D + 3) / 2)
        self.K = 5 # output  3or5
        self.M = 3 
        self.epsilon = 0.01
        self.batch_size = 1
        self.max_epoch = 600
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
        tmp = [
            input_data[i] * input_data[j]
            for i in range(self.D)
            for j in range(i, self.D)
        ]
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

        return O3

    def backward(self, batch_T):
        grad = [
            (
                (self.IO["O3"][i] - batch_T[i]).reshape(self.K, 1)
                * (self.IO["O2"][i] / self.IO["O3"][i].reshape(self.K, 1))
            ).reshape(self.K, self.M, 1)
            * self.IO["I1"][i]
            for i in range(self.batch_size)
        ]
        self.grad = np.sum(grad, axis=0) / self.batch_size
        self.weight -= self.epsilon * self.grad

    def train(self, data, label):
        for epoch in range(self.max_epoch):
            iter_per_epoch = max(int(len(data) / self.batch_size), 1)
            for i in range(iter_per_epoch):
                idx = np.random.choice(len(data), size=self.batch_size, replace=False)
                _ = self.forward(data[idx])
                self.backward(label[idx])
            Y = self.forward(data)
            loss = np.sum((Y - label) ** 2)
            entropy = np.sum(-1 * label * np.log(Y)) / len(data)
            acc = sum(np.argmax(t) == np.argmax(y) for t, y in zip(label, Y)) / len(
                data
            )
            print(epoch, ":", acc, loss, entropy)
        return Y

    def test(self, data, label):
        Y = self.forward(data)
        acc = sum(np.argmax(t) == np.argmax(y) for t, y in zip(label, Y)) / len(data)
        print("Accurcy : ", acc)
        return Y


if __name__ == "__main__":
    data_train = np.loadtxt("llgmn/examinee-nakaoka/5move/5movement_data_train.csv", delimiter=",")
    label_train = np.loadtxt("llgmn/examinee-nakaoka/5move/5movement_label_train.csv", delimiter=",")
    data_test = np.loadtxt("llgmn/examinee-nakaoka/5move/5movement_data_test.csv", delimiter=",")
    label_test = np.loadtxt("llgmn/examinee-nakaoka/5move/5movement_label_test.csv", delimiter=",")

    ll = LLGMN()
    Y_train = ll.train(data_train, label_train)
    Y_test = ll.test(data_test, label_test)

    ll.save_weight("llgmn/examinee-nakaoka/5move/weights2.npy")
