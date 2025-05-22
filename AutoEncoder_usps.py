
import numpy as np
from sklearn.model_selection import train_test_split
from data import load_usps_from_pkl
from modules import Sequential, Linear, Tanh
from losses import MSELoss
from optimizer import SGD
import matplotlib.pyplot as plt
from Evalu_Visu import plot_reconstruction
from sklearn.preprocessing import StandardScaler

# === Chargement des données
X, y = load_usps_from_pkl("usps.pkl")
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=0)

# === Normalisation Standard
scaler = StandardScaler()
scaler.fit(X_train*255.0)
X_train_scaled = scaler.transform(X_train*255.0)
X_test_scaled = scaler.transform(X_test*255.0)

# === Auto-encodeur : Encodeur + Décodeur
encodeur = Sequential(
    Linear(256, 128, init_type=1),
    Tanh(),
    Linear(128, 64, init_type=1),
    Tanh(),
    Linear(64, 48, init_type=1),
    Tanh()
)

decodeur = Sequential(
    Linear(48, 64, init_type=1),
    Tanh(),
    Linear(64, 128, init_type=1),
    Tanh(),
    Linear(128, 256, init_type=1),
    Tanh()
)

autoencoder = Sequential(*encodeur.modules, *decodeur.modules)

# === Apprentissage
loss_fn = MSELoss()
#print("Avant entraînement :")
#print(autoencoder.modules[0]._parameters[:5])

losses = SGD(autoencoder, loss_fn, X_train_scaled, X_train_scaled, eps=0.01, batch_size=64, n_epochs=200)

#print("Après entraînement :")
#print(autoencoder.modules[0]._parameters[:5])

# === Courbe de coût
plt.plot(losses)
plt.title("Descente du coût")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# === Variance des codes
code = encodeur.forward(X_test_scaled)
#print("Variance des codes :", np.var(code, axis=0))

# === Reconstruction et visualisation
X_rec = autoencoder.forward(X_test_scaled)
X_rec_denorm = scaler.inverse_transform(X_rec)
X_orig_denorm = scaler.inverse_transform(X_test_scaled)

plot_reconstruction(X_orig_denorm, X_rec_denorm, n=10)
