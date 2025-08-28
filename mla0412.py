import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def gradient_descent(X, y, lr=0.1, epochs=100):
    m = len(y)
    theta0, theta1 = 0, 0
    losses = []
    for _ in range(epochs):
        y_pred = theta0 + theta1 * X
        d0 = -(2/m) * sum(y - y_pred)
        d1 = -(2/m) * sum((y - y_pred) * X)
        theta0 -= lr * d0
        theta1 -= lr * d1
        loss = np.mean((y - y_pred)**2)
        losses.append(loss)
    return theta0, theta1, losses

# 21. Actual Data (Iris)
iris = load_iris()
X1 = iris.data[:,0]   # sepal length
y1 = iris.data[:,2]   # petal length
t0, t1, loss1 = gradient_descent(X1, y1)

print("Iris Data Theta:", t0, t1)
plt.plot(loss1, label="Iris Data Loss")
plt.legend(); plt.show()

# 22. Modified Data (Synthetic)
np.random.seed(0)
X2 = np.random.rand(50) * 10
y2 = 5 + 2 * X2 + np.random.randn(50)
t0, t1, loss2 = gradient_descent(X2, y2)

print("Modified Data Theta:", t0, t1)
plt.plot(loss2, label="Modified Data Loss", color="red")
plt.legend(); plt.show()
