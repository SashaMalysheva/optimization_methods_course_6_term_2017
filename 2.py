import random
import numpy as np

import numpy as np

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
    print(x)



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

    print(x)

def is_singular(m):
    return np.any(np.diag(m) == 0)

def qr(A):
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
        H = np.eye(m)
        H[i:, i:] = make_householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
    return Q, A


def make_householder(a):
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.dot(v[:, None], v[None, :])
    return H


def polyfit(x, y, n):
    return lsqr(x[:, None] ** np.arange(n + 1), y.T)


def lsqr(a, b):
    q, r = qr(a)
    _, n = r.shape
    return np.linalg.solve(r[:n, :], np.dot(q.T, b)[:n])

#v = random.randint(10, 20)
v = 3
g = [[random.randint(0, 1) for j in range(v)] for i in range(v)]
g = [[0,1,0],[1,0,1],[0,1,0]]
#for i in range(v):
#    for j in range(i, v):
#        g[i][j] = g[j][i]
m = np.zeros((v, v + 1))
for i in range(v):
    for j in range(v):
        if sum(g[i]) != 0:
            m[i, j] = g[i][j] * 1 / sum(g[i])
    # p1*e1+p2*e2+...+pn*en=pi
    m[i, i] -= 1

m[0] += 1
solve_gauss(m)

