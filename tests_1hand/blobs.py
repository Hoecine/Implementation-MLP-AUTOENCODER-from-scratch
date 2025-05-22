
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# === Import de tes classes ===
from modules import Sequential, Linear, Tanh, Sigmoide
from losses import BCELoss
from optimizer import SGD 
from Evalu_Visu import plot_decision_boundary

def make_mixed_blobs(n_samples=3000, centers=6, cluster_std=1.2, random_state=42):
    X, labels = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state
    )
    
    # Attribution binaire : moitié des centres → classe 0, autre moitié → classe 1
    y = (labels >= centers // 2).astype(int).reshape(-1, 1)
    
    return X, y

X, y = make_mixed_blobs(n_samples=3000, centers=6, cluster_std=1.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

net = Sequential(
    Linear(2, 64),
    Tanh(),
    Linear(64, 128),
    Tanh(),
    Linear(128, 64),
    Tanh(),
    Linear(64, 1),
    Sigmoide()
)


loss = BCELoss()
losses = SGD(net, loss, X_train, y_train, eps=0.15, batch_size=32, n_epochs=300)

y_pred = net.forward(X_test)
acc_test = np.mean((y_pred > 0.5).astype(int) == y_test)
acc_train = np.mean((net.forward(X_train) > 0.5).astype(int) == y_train)

print(f"Accuracy MLP en train : {acc_train * 100:.2f}%")
print(f"Accuracy MLP en test  : {acc_test * 100:.2f}%")

plot_decision_boundary(net, X_test, y_test, "MLP - Frontière de décision")
