import numpy as np
np.random.seed(1002)


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


def create_random_vector(n, l=1, r=10):
    return np.random.random(n) * (r - l) + l


def ordinary_least_squares(x, y, deg):
    x_powers = np.array([np.sum(x ** i) for i in range(2 * deg + 1)])
    matrix = np.array([[x_powers[i + j] for i in range(deg + 1)] for j in range(deg + 1)])
    b = np.array([np.sum(x ** i * y) for i in range(deg + 1)])
    coefficients = np.linalg.solve(matrix, b)
    print("o", np.sum(abs(matrix.dot(coefficients) - b)))
    return np.poly1d(coefficients[::-1])


def cholesky_solve(A, b):
    L = cholesky1(A.T.dot(A))
    # A.T*Ax=A.T*b => L*L.T x = A.T*b => Lw=A.Tb => L.Tx=w
    print("delta: cholesky", np.sum(L.dot(L.T) - A.dot(A)))
    w = solve(L, A.T.dot(b).reshape((len(A.T.dot(b)), 1)))
    ans = solveT(L.T, w.reshape(len(w), 1))
    print("delta ansver: ", np.sum(A.dot(ans) - b))
    return ans


def cholesky_least_squares(x, y, deg):
    x_powers = np.array([np.sum(x ** i) for i in range(2 * deg + 1)])

    A = np.array([[x_powers[i + j] for i in range(deg + 1)] for j in range(deg + 1)])
    b = np.array([np.sum(x ** i * y) for i in range(deg + 1)])

    ans = cholesky_solve(A, b)
    return np.poly1d(ans[::-1])


m = 8
n = 4
x = create_random_vector(m)
y = create_random_vector(m)

poly_ols = ordinary_least_squares(x, y, n)
y_ols = np.array([poly_ols(x[i]) for i in range(m)])
print(np.sum(y_ols-y))

poly_cholesky = cholesky_least_squares(x, y, n)
y_cholesky = np.array([poly_cholesky(x[i]) for i in range(m)])
print(np.sum(y_cholesky-y))



