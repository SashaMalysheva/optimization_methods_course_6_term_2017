import random
import numpy as np
#http://math.volchenko.com/Lectures/MNK.pdf


def bubble_max_row(m, k):
    ind = k + np.argmax(np.abs(m[k:, k]))
    if ind != k:
        m[k, :], m[ind, :] = np.copy(m[ind, :]), np.copy(m[k, :])


def solve_gauss(m):
    n = m.shape[0]
    # forward trace
    for k in range(n - 1):
        bubble_max_row(m, k)
        for i in range(k + 1, n):
            frac = m[i, k] / m[k, k]
            m[i, :] -= m[k, :] * frac

    if is_singular(m):
        print('The system has infinite number of answers...')
        return

    # backward trace
    x = np.matrix([0.0 for i in range(n)]).T
    for k in range(n - 1, -1, -1):
        x[k, 0] = (m[k, -1] - m[k, k:n] * x[k:n, 0]) / m[k, k]
    return x


def is_singular(m):
    return np.any(np.diag(m) == 0)

m = random.randint(6, 7)
n = random.randint(3, 5)
x = np.array([random.randint(1, 10) for i in range(m)])
y = np.array([random.randint(1, 10) for i in range(m)])
F = [[x[i] ** j for j in range(n)] for i in range(m)]
F = np.array(F)

# F^TF x = F^TY
m = np.zeros((n, n+1))
m[:, :-1] = np.matmul(F.transpose(), F)
m[:, n] = np.matmul(F.transpose(), y)

x = np.array(solve_gauss(m)).flatten()
result = sum((np.sum(F * x, axis=1) - y.T) ** 2)
print(result)