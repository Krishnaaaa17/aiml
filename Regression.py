import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
def lowess(x, y, f, degree=1, gamma=None):
    if gamma is None:
        gamma = 1.0 / (2.0 * f**2)
    model = make_pipeline(PolynomialFeatures(degree), KernelRidge(kernel='rbf', gamma=gamma))
    model.fit(x[:, np.newaxis], y)
    yest = model.predict(x[:, np.newaxis])
    return yest
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + 0.3 * np.random.randn(100)
smoothed_y = lowess(x, y, f=0.5, degree=1, gamma=None)
plt.plot(x, y, "r.", label='Original Data')
plt.plot(x, smoothed_y, "b-", label='LOWESS Fit') 
plt.legend()
plt.show()
