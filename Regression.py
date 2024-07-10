import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def lowess(x, y, f, iterations):
    n, r = len(x), int(np.ceil(f * len(x)))
    h = np.array([np.partition(np.abs(x - x[i]), r)[r] for i in range(n)])
    w = (1 - np.clip(np.abs((x[:, None] - x) / h[:, None]), 0, 1) ** 3) ** 3
    yest, delta = np.zeros(n), np.ones(n)
    for _ in range(iterations):
        for i in range(n):
            weights = delta * w[:, i]
            W = np.diag(weights)  # Create a diagonal matrix of weights
            A = np.array([np.ones(n), x]).T
            b = y
            theta = linalg.solve(A.T @ W @ A, A.T @ W @ b)
            yest[i] = theta[0] + theta[1] * x[i]
        delta = (1 - np.clip((y - yest) / (6 * np.median(np.abs(y - yest))), -1, 1) ** 2) ** 2
    return yest

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + 0.3 * np.random.randn(100)
plt.plot(x, y, "r.", label='Original Data')  # Plot original data points
plt.plot(x, lowess(x, y, 0.25, 3), "b-", label='LOWESS Fit')  # Plot LOWESS smoothed curve
plt.legend()
plt.show()