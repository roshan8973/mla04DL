import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np
iris = load_iris()
X = iris.data[:100, :2]  
y = iris.target[:100]    

model = LogisticRegression()
model.fit(X, y)
plt.scatter(X[y==0, 0], X[y==0, 1], color="blue", label="Setosa")
plt.scatter(X[y==1, 0], X[y==1, 1], color="red", label="Versicolor")
x_vals = np.linspace(X[:,0].min(), X[:,0].max(), 100)
y_vals = -(model.coef_[0][0] * x_vals + model.intercept_[0]) / model.coef_[0][1]
plt.plot(x_vals, y_vals, color="green", label="Decision Boundary")

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Linear Separability (Iris Setosa vs Versicolor)")
plt.legend()
plt.show()
