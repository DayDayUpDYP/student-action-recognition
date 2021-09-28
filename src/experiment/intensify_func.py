import numpy as np
import torch
from matplotlib import pyplot as plt


def intensify_func(x, n, decay_rate):
    return x + decay_rate * (x - 1. / n)


def origin_func(x):
    return x * x


def update(y, n):
    res = []
    for yi in y:
        res.append(intensify_func(yi, n, 0.5))
    return res


if __name__ == '__main__':
    N = 9
    STEP = 5

    x = np.linspace(-1, 1, N)
    y = np.array([origin_func(i) for i in x])
    y = torch.nn.Softmax(0)(torch.Tensor(y)).numpy()

    print(x)
    print(y)
    plt.axhline(1 / N)

    l = []
    titles = []

    # L1 = plt.scatter(x, y)
    for epoch in range(STEP):
        L, = plt.plot(x, y)
        y = update(y, N)
        l.append(L)
        titles.append(f'{epoch}')
        plt.legend(l, titles, loc='upper right')
    plt.show()
