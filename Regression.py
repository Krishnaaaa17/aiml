import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def lowess_regression(x, y, f, degree=1):
    model = make_pipeline(PolynomialFeatures(degree), HuberRegressor(epsilon=1.35))

    model.fit(x[:, np.newaxis], y)
    yest = model.predict(x[:, np.newaxis])

    return yest

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + 0.3 * np.random.randn(100)

plt.plot(x, y, "r.", label='Original Data')  # Plot original data points
plt.plot(x, lowess_regression(x, y, 0.25, degree=1), "b-", label='Regression Fit')  # Plot regression curve
plt.legend()
plt.show()
