from math import sqrt
import numpy as np
from numpy.linalg import cholesky


def solve(L, v):
    b = np.copy(v)
    v = np.zeros(L.shape[0])
    for i in range(len(v)):
        for k in range(i):
            b[i] -= v[k] * L[i, k]
        v[i] = b[i] / L[i, i]
    return v


def solveT(L, v):
    b = np.copy(v)
    v = np.zeros(L.shape[0])
    for i in range(len(v) - 1, -1, -1):
        for k in range(i + 1, L.shape[0]):
            b[i] -= v[k] * L[i, k]
        v[i] = b[i] / L[i, i]
    return v


def cholesky1(A):
    n = A.shape[0]
    L = np.zeros(A.shape)
    for row in range(n):
        for col in range(row + 1):
            tmp_sum = 0.0
            for j in range(col):
                tmp_sum += L[row, j] * L[col, j]
            if row == col:
                L[row, col] = np.sqrt(A[row, row] - tmp_sum)
            else:
                L[row, col] = (1.0 / L[col, col] * (A[row, col] - tmp_sum))
        L[row, row + 1:] = 0.0
    return L

def gen_simm_pol_matrix(n, m, l=1, r=10):
    m = np.random.rand(n, m) * (r - l) + l
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m[j, i]
    A = m.dot(np.transpose(m))
    assert (np.all(np.linalg.eigvals(A) > 0))
    return A

def create_random_vector(n, l=1, r=4):
    return np.random.random(n) * (r - l) + l


y = create_random_vector(4)
x = create_random_vector(4)
A = np.array([x ** i for i in range(4)]).T
b = y
L = cholesky(A.dot(A))
print(np.sum(A.dot(A) - L.dot(L.T)))
# A*Ax=A*b => L*L.T x = A*b => Lw=Ab => L.Tx=w
w = solve(L, A.dot(b).reshape((len(A.dot(b)), 1)))
ans = solveT(L.T, w.reshape(len(w), 1))
print(A.dot(b), L.dot(L.T.dot(ans)))
print(A.dot(ans), b)
