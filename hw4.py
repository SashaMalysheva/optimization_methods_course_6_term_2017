import math
import time


def measureTime(a, f, y):
    start = time.clock()
    a(f, iteration=10 ** y)
    elapsed = time.clock()
    elapsed = elapsed - start
    print("Time spent in  is: ", elapsed)


def derivative(f):
    def compute(x, dx=0.0000001):
        return (f(x + dx) - f(x)) / dx
    return compute


def newtons_method(f, iteration=10 ** 10,  x=1, tolerance=100 ** -10000):
    df = derivative(f)
    num = 0
    while True:
        num += 1
        if df(x) != 0:
            x1 = x - f(x) / df(x)
            t = abs(f(x))
            if t < tolerance:
                break
            if num == iteration:
                break
            x = x1
        else:
            break
    return x


def f1(x):
    return x ** 4 - 3 * x ** 2 + 75 * x - 10000


def f2(x):
    return math.tan(x) - x


def f3(x):
    return x ** 5 - x - 0.2


def f4(x):
    return math.sin(x)


def f5(x):
    return x ** 2 - 30000


def print_result(f, iteration=10 ** 4, x_approx=1, tolerance=100 ** -10000):
    x = newtons_method(f, iteration, x_approx, tolerance)
    print("x = %0.5f" % x)
    print("Testing")
    print("%0.10f" % (f(x)))
    print()


print('Function x ** 4 - 3 * x ** 2 + 75 * x - 10000')
print_result(f1,  tolerance=0.000001, x_approx=-10)
print('Function tan(x) - x')
print_result(f2, tolerance=0.000005, x_approx=4.4)
print('Function x ** 5 - x - 0.2')
print_result(f3, tolerance=0.000001)
print('Function sin(x)')
print_result(f4, tolerance=0.000001, x_approx=3)

for y in range(8):
    print('Function x ** 1/2 with '+str(10**y)+' iteration')
    measureTime(newtons_method, f5, y)
    print()

