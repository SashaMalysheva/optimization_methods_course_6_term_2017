import numpy as np
np.random.seed(105)


def subgradient_metod(f, prec):
    cnt_op = 1
    x_new = np.ones(f.n)
    while abs(np.sum(f.evaluate(x_new))/1000) > prec:
        cnt_op += 1
        alpha = 1 / cnt_op
        x_old = x_new
        g = f.find_gradient(x_new)
        g -= np.sign(g) * 0.001
        x_new = x_old - alpha * g
        #print("error: ", abs(np.sum(f.evaluate(x_new))/1000), " steps: ",cnt_op)
    return x_new, cnt_op


def fast_subgradient_metod(f, x0, prec):
    cnt_op = 1
    x_new = np.ones(f.n)
    while abs(np.sum(f.evaluate(x_new))/1000) > prec:
        cnt_op += 1
        x_old = x_new
        g = f.find_gradient(x_new)
        g -= np.sign(g) * 0.001
        alpha = abs((f.evaluate(x_new) - f.evaluate(x0)) / np.sum(g))
        x_new = x_old - alpha * g
        print("alpha: ", alpha, " error: ", abs(np.sum(f.evaluate(x_new))/1000), " steps: ",cnt_op)
    return x_new, cnt_op


def gen_simm_pol_matrix(n, m, l=0, r=1):
    A = np.random.rand(m, n) * (r - l) + l
    return A


class function:
    def __init__(self, n=20, m=6):
        self.n = n
        # number of variables
        self.m = m
        self.A = gen_simm_pol_matrix(n, m)
        self.b = gen_simm_pol_matrix(1, m)
        print('A\n', self.A)
        print('b\n', self.b)

    def evaluate(self, x):
        return np.max(np.reshape(np.sum(np.multiply(self.A, x), axis=1), (self.m, 1)) + self.b)

    def find_gradient(self, x):
        return self.A[np.argmax(np.reshape(np.sum(np.multiply(self.A, x), axis=1), (self.m, 1)) + self.b)]


f = function()
x_new, cnt_op1 = subgradient_metod(f, prec=0.00001)
x_new, cnt_op0 = fast_subgradient_metod(f, x_new, prec=0.00001)
print("subgradient_metod - answer: ", x_new, " steps:", cnt_op0)
print("fast_subgradient_metod - answer: ", x_new, " steps:", cnt_op1)
