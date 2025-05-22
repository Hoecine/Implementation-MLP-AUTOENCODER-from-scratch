
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# === Import de tes classes ===
from modules import Sequential, Linear, Tanh, Sigmoide
from losses import BCELoss
from optimizer import SGD  

np.random.seed(42)
# === Données de l’échiquier ===
def generate_checkerboard(n_samples=3000, k=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = np.random.rand(n_samples, 2)
    y = (((X[:, 0] * k).astype(int) + (X[:, 1] * k).astype(int)) % 2).astype(int)
    return X, y.reshape(-1, 1)

# === Visualisation ===
def plot_decision_boundary(model, X, y, title):
    h = 0.01
    xx, yy = np.meshgrid(np.arange(0, 1, h), np.arange(0, 1, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.forward(grid)
    Z = (Z > 0.5).astype(int).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    plt.show()

# === Chargement des données ===
X, y = generate_checkerboard(n_samples=3000, k=10, seed=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# === Réseau Perceptron ===
net1 = Sequential(
    Linear(2, 1),
    Sigmoide()
)
loss1 = BCELoss()
losses1 = SGD(net1, loss1, X_train, y_train, eps=0.01, batch_size=32, n_epochs=90)
y_pred1 = net1.forward(X_test)
acc1 = np.mean((y_pred1 > 0.5).astype(int) == y_test)
print(f"Accuracy Perceptron : {acc1 * 100:.2f}%")

# === Réseau MLP Profond ===
net2 = Sequential(
    Linear(2, 50),
    Tanh(),
    Linear(50, 100),
    Tanh(),
    Linear(100, 50),
    Tanh(),
    Linear(50, 1),
    Sigmoide()
)
loss2 = BCELoss()
losses2 = SGD(net2, loss2, X_train, y_train, eps=0.09, batch_size=10, n_epochs=2000)
y_predtr = net2.forward(X_train);acctr = np.mean((y_predtr > 0.5).astype(int) == y_train);print(f"Accuracy MLP en train : {acctr * 100:.2f}%")

y_pred2 = net2.forward(X_test)
acc2 = np.mean((y_pred2 > 0.5).astype(int) == y_test)
print(f"Accuracy MLP en test : {acc2 * 100:.2f}%")

# === Affichage des courbes de coût ===
plt.plot(losses1, label="Perceptron")
plt.plot(losses2, label="MLP")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Courbes de coût")
plt.legend()
plt.grid(True)
plt.show()

# === Frontières de décision ===
plot_decision_boundary(net1, X_test, y_test, "Perceptron - Frontière de décision")
plot_decision_boundary(net2, X_test, y_test, "MLP - Frontière de décision")
