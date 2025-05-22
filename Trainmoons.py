

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from modules import Sequential, Linear, Tanh, LogSoftmax
from losses import NLLLoss
from optimizer import SGD

# === Données non linéaires ===
X, y = make_moons(n_samples=1000, noise=0.4, random_state=0)
y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# === Modèle 1 : perceptron simple ===
perceptron = Sequential(
    Linear(2, 2),
    LogSoftmax()
)

loss = NLLLoss()
losses_p = SGD(perceptron, loss, X_train, y_train, eps=0.1, batch_size=16, n_epochs=100)

# === Modèle 2 : MLP ===
mlp = Sequential(
    Linear(2, 10),
    Tanh(),
    Linear(10, 2),
    LogSoftmax()
)

losses_mlp = SGD(mlp, loss, X_train, y_train, eps=0.08, batch_size=16, n_epochs=100)

# === Accuracy des deux modèles ===
def accuracy(model, X, y):
    y_pred = model.forward(X)
    pred = np.argmax(y_pred, axis=1)
    true = np.argmax(y, axis=1)
    return np.mean(pred == true)

print(f"Accuracy Perceptron en train : {accuracy(perceptron, X_train, y_train) * 100:.2f}%");print(f"Accuracy Perceptron en test: {accuracy(perceptron, X_test, y_test) * 100:.2f}%")
print(f"Accuracy MLP en train       : {accuracy(mlp, X_train, y_train) * 100:.2f}%");print(f"Accuracy MLP en test       : {accuracy(mlp, X_test, y_test) * 100:.2f}%")

# === Visualisation de la frontière de décision ===
def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=np.argmax(y, axis=1), edgecolor='k')
    plt.title(title)
    plt.show()

plot_decision_boundary(perceptron, X_test, y_test, "Perceptron - Données non linéaires")
plot_decision_boundary(mlp, X_test, y_test, "MLP - Données non linéaires")