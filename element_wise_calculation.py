import numpy as np


def calculate(R, x):
    gamma = 2
    return np.sum(np.exp(-gamma * np.square(R-x)))

R = np.array([[1, 2], [3, 4]])
x = np.array([5, 6])

result = np.sum([calculate(R, e) for e in x])


