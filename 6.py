import numpy as np
import random


def solve(b, c, a=1):
    d = b ** 2 - 4 * a * c
    if d >= 0:
        return max((-b + np.sqrt(d)) / 2 * a, (-b - np.sqrt(d)) / 2 * a)
    elif d < 0:
        print("here")
        return 0.1


def find_grad_simple(A, b, prec):
    cnt_op = 0
    x_new = np.zeros((A.shape[0],))
    r = A.dot(x_new) - b

    while np.sum(r ** 2) > prec:
        cnt_op += 1
        x_old = x_new
        r = A.dot(x_old) - b
        alpha = r.dot(r) / r.dot(A.dot(r))
        x_new = x_old - alpha * r
        # print("alpha: ", alpha, " r: ", np.sum(r), " intr: ", cnt_op)
    return x_new, cnt_op


def free_grad(A, b, prec):
    cnt_op = 0
    x_new = np.zeros((A.shape[0],))
    r = A.dot(x_new) - b
    p = np.zeros((A.shape[0],))
    while np.sum(r ** 2) > prec:
        j = random.randrange(A.shape[0])
        p[j] = 1
        cnt_op += 1
        x_old = x_new
        r = A.dot(x_old) - b
        alpha = r.dot(p) / p.dot(A.dot(p))
        x_new = x_old - alpha * p
        p[j] = 0
        # print("alpha: ", alpha, " r: ", np.sum(r), " intr: ", cnt_op)
    return x_new, cnt_op


def opt_grad(A, b, prec):
    cnt_op = 0
    x_new = np.zeros((A.shape[0])) + 1
    y = np.copy(x_new)
    w, v = np.linalg.eig(A)
    M = np.max(w)
    m = np.min(w)
    a_new = 0.5
    r = A.dot(x_new) - b
    while np.sum(r ** 2) > prec:
        cnt_op += 1
        r = A.dot(x_new) - b
        x_old = x_new
        x_new = y - 1 / M * (A.dot(y) - b)
        a_old = a_new
        a_new = solve(-m / M + a_old ** 2, -a_old ** 2)
        y = x_new + (a_old * (1 - a_old)) / (a_old ** 2 + a_new) * (x_new - x_old)
        # print(" error: ", np.sum(r**2), " intr: ", cnt_op)
    return x_new, cnt_op


def conjugate_grad(A, b, prec):
    cnt_op = 0
    x_new = np.zeros((A.shape[0],))
    p = r_new = b
    while np.sum(r_new ** 2) > prec:
        cnt_op += 1
        x_old = x_new
        r_old = r_new

        alpha = r_old.dot(r_old) / (p.dot(A.dot(p)))
        x_new = x_old + alpha * p

        r_new = b - A.dot(x_new)
        beta = r_new.dot(r_new) / (r_old.dot(r_old))
        p = r_new + beta * p
        # print("alpha: ", alpha, " r: ", np.sum(r_new), " intr: ", cnt_op)
    return x_new, cnt_op


def gen_simm_pol_matrix(n, l=0, r=1):
    m = np.random.rand(n, n) * (r - l) + l
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m[j, i]
    A = m.dot(np.transpose(m))
    print("A\n", A)
    assert (np.all(np.linalg.eigvals(A) > 0))
    return A


def create_random_vector(n, l=0, r=1):
    return np.random.random(n) * (r - l) + l


if __name__ == "__main__":
    n = 5
    A = gen_simm_pol_matrix(n)
    x = create_random_vector(n)
    b = np.dot(A, x)

    prec = 0.00001

    print("opt_grad:")
    ans0, cnt_op0 = opt_grad(A, b, prec)
    print("simple_grad:")
    ans1, cnt_op1 = find_grad_simple(A, b, prec)
    print("free_grad:")
    ans2, cnt_op2 = free_grad(A, b, prec)
    print("conhugate_grad:")
    ans4, cnt_op4 = conjugate_grad(A, b, prec)

    print("opt_grad:", np.sum((ans0 - x) ** 2), cnt_op0)
    print("simple_grad:", np.sum((ans1 - x) ** 2), cnt_op1)
    print("free_grad:", np.sum((ans2 - x) ** 2), cnt_op2)
    print("conjugate_grad:", np.sum((ans4 - x) ** 2), cnt_op4)

    n = 5
    A = gen_simm_pol_matrix(n)
    y = create_random_vector(n)
    x = create_random_vector(n)
    b = np.dot(A, x)

    # (A*A.T) * lambda = 2(Ay-b)
    A = A.dot(A.T)
    b = 2 * (A.dot(y) - b)
    print(A, b)
    prec = 0.00001
    ans0, cnt_op0 = opt_grad(A, b, prec)
    ans1, cnt_op1 = find_grad_simple(A, b, prec)
    ans2, cnt_op2 = free_grad(A, b, prec)
    ans4, cnt_op4 = conjugate_grad(A, b, prec)

    # x = y- 1/2 * A.T * lambda
    x1 = y - 1 / 2 * A.T.dot(ans1)
    x2 = y - 1 / 2 * A.T.dot(ans2)
    x4 = y - 1 / 2 * A.T.dot(ans4)

    print("opt_grad:", np.sum((ans0 - x1) ** 2), cnt_op0)
    print("simple_grad:", np.sum((ans1 - x1) ** 2), cnt_op1)
    print("free_grad:", np.sum((ans2 - x2) ** 2), cnt_op2)
    print("conhugate_grad:", np.sum((ans4 - x4) ** 2), cnt_op4)
