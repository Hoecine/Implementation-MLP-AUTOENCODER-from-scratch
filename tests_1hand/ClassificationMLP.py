# ======= Classification avec MLP =======
import numpy as np
from sklearn.model_selection import train_test_split
from data import load_usps_from_pkl
from modules import Sequential, Linear, Tanh, LogSoftmax, ReLU
from losses import NLLLoss
from optimizer import SGD
import matplotlib.pyplot as plt


X, y = load_usps_from_pkl("usps.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Définition du modèle 
# Modèle 1 (1 couche cachée seulement)
net = Sequential(
    Linear(256, 128),
    Tanh(),
    Linear(128, 10),
    LogSoftmax()
)

loss = NLLLoss()
losses = SGD(net, loss, X_train, y_train, eps=0.01, batch_size=30, n_epochs=300)

# Evaluation su test
y_predtr = net.forward(X_train)
y_predts = net.forward(X_test) 

#classes prédites sur train
pred_labelstr = np.argmax(y_predtr, axis=1)
true_labelstr = np.argmax(y_train, axis=1)
accuracytr = np.mean(pred_labelstr == true_labelstr)
print(f"Accuracy sur train set : {accuracytr * 100:.2f}%")

#Classes prédites pour le test
pred_labels = np.argmax(y_predts, axis=1)
true_labels = np.argmax(y_test, axis=1)

accuracyts = np.mean(pred_labels == true_labels)
print(f"Accuracy sur test set : {accuracyts * 100:.2f}%")

# Courbe de perte
plt.figure(figsize=(6, 3))
plt.plot(losses, label="Loss NLL")
plt.xlabel("Époque")
plt.ylabel("Perte moyenne")
plt.title("Évolution de la perte durant l'apprentissage")
plt.grid()
plt.legend()

"""
# Modèle 2 (2 couches cachées) 
net1 = Sequential(
    Linear(256, 128),
    Tanh(),
    Linear(128, 64),
    ReLU(),
    Linear(64, 10),
    LogSoftmax()
)
loss1 = NLLLoss()
losses1 = SGD(net1, loss1, X_train, y_train, eps=0.005, batch_size=32, n_epochs=80)
y_pred1 = net1.forward(X_test)
pred_labels1 = np.argmax(y_pred1, axis=1)
true_labels1 = np.argmax(y_test, axis=1)
accuracy1 = np.mean(pred_labels1 == true_labels1)
print(f"Accuracy sur test set : {accuracy1 * 100:.2f}%")

plt.figure(figsize=(6, 3))
plt.plot(losses1, label="Loss NLL")
plt.xlabel("Époque")
plt.ylabel("Perte moyenne")
plt.title("Évolution de la perte durant l'apprentissage")
plt.grid()
plt.legend()
plt.show()
"""