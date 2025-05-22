
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from modules import Sequential, Linear
from losses import MSELoss
from optimizer import SGD


# Génération des données 
np.random.seed(0)
x = np.linspace(-1, 1, 200).reshape(-1, 1)
noise = np.random.normal(0, 0.8, size=x.shape) #ajout de bruit gaussier, pour simuler le bruit dans les données, suivant une loi normale autour d'une droite qu'on cherche a approcher
y = 3 * x + 2 + noise
"""
# Réseau - Perceptron

net = Sequential(
    Linear(1,1)
)

#Perte aux moindres carrés

loss = MSELoss()

# Apprentissage

losses = SGD(net, loss, x, y, eps=0.1, batch_size=10, n_epochs=100)

# Prédiction

y_pred = net.forward(x)

# Résultat - Régression Linéaire

plt.figure(figsize=(6, 4))
plt.scatter(x, y, label="Nuage de points", alpha=0.5)
plt.plot(x, y_pred, label="Prédiction du modèle", color='red')
plt.title("Régression linéaire : y = 3x + 2 + bruit")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(6, 3))
plt.plot(losses, label="MSE par epoch")
plt.xlabel("Époque")
plt.ylabel("MSE moyenne")
plt.title("Évolution de la perte MSE durant l'apprentissage")
plt.grid()
plt.legend()
plt.show()
"""

# Partie avec Test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
net1 = Sequential(
    Linear(1,1)
)
loss = MSELoss()
losses = SGD(net1, loss, x_train, y_train, eps=0.1, batch_size=10, n_epochs=100)

y_predtr = net1.forward(x_train)

#Visualisation sur les données train
plt.figure(figsize=(6, 4))
plt.scatter(x_train, y_train, label="Nuage de points", alpha=0.5)
plt.plot(x_train, y_predtr, label="Prédiction du modèle", color='red')
plt.title("Régression linéaire : y = 3x + 2 + bruit")
plt.legend()
plt.grid()


plt.figure(figsize=(6, 3))
plt.plot(losses, label="MSE par epoch")
plt.xlabel("Époque")
plt.ylabel("MSE moyenne")
plt.title("Évolution de la perte MSE durant l'apprentissage")
plt.grid()
plt.legend()


#visualisation sur les données test
y_predts = net1.forward(x_test)
mse_test = np.mean((y_test - y_predts)**2)
print(f"MSE sur le test set : {mse_test:.4f}")

plt.figure(figsize=(4, 4))
plt.scatter(x_test, y_test, label="données test", alpha=0.6)
plt.plot(x_test, y_predts, label= "Prédiction du modèle", color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

#il me reste de tesla prediction sur tout x, et aprés comparer a la vraie droite non bruitée!! 

