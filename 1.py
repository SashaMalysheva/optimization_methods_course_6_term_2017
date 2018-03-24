import random
import numpy as np


def bubble_max_row(m, k):
    ind = k + np.argmax(np.abs(m[k:, k]))
    if ind != k:
        m[k, :], m[ind, :] = np.copy(m[ind, :]), np.copy(m[k, :])


def solve_with_rotation(m):
    n = m.shape[0]
    # forward trace

    for i in range(n - 1):
        for j in range(i + 1, n):
            c = m[i, i] / (m[i, i] ** 2 + m[j, i] ** 2) ** .5
            s = m[j, i] / (m[i, i] ** 2 + m[j, i] ** 2) ** .5
            tmp1 = m[i, :] * c + m[j, :] * s
            tmp2 = m[i, :] * -s + m[j, :] * c
            m[i, :] = tmp1
            m[j, :] = tmp2
    if is_singular(m):
        print('The system has infinite number of answers...')
        return

    x = np.matrix([0.0 for i in range(n)]).T
    for k in range(n - 1, -1, -1):
        x[k, 0] = (m[k, -1] - m[k, k:n] * x[k:n, 0]) / m[k, k]

    return x


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

v = random.randint(10, 11)
g = np.array([[random.randint(0,1) for j in range(v)] for i in range(v)])
print(g)
m = np.zeros((v, v + 1))

for i in range(1, v):
    for j in range(v):
        m[i, j] = -1*g[j][i]
    m[i, i] = sum(g[i])
m[0, 0] = 1
m[0, v] = 1
x = solve_with_rotation(m)

for i in range(v):
    s = 0
    for j in range(v):
        s += int(g[i,j]) * int(x[i])
        s -= int(g[i,j]) * int(x[i])
    print(s)