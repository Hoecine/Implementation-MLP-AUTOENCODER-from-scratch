import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y, title):
    h = 0.1  # résolution de la grille
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.forward(grid)
    if Z.ndim > 1:
        Z = Z.ravel()
    Z = (Z > 0.5).astype(int)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    plt.show()


def plot_reconstruction(X, X_rec, n=10):
    plt.figure(figsize=(n, 2))
    s = int(np.sqrt(X.shape[1]))
    for i in range(n):
        # Originale
        plt.subplot(2, n, i + 1)
        plt.imshow(X[i].reshape(s, s), cmap="gray")
        plt.axis('off')
        
        # Reconstituée
        plt.subplot(2, n, n + i + 1)
        plt.imshow(X_rec[i].reshape(s, s), cmap="gray")
        plt.axis('off')
    plt.show()
    
def evaluate(model, X, y_true):
    y_pred = model.forward(X)
    predicted = np.argmax(y_pred, axis=1)
    return np.mean(predicted == y_true) * 100


from scipy.ndimage import rotate
from sklearn.utils import shuffle

def augment_and_noisy_images(X, y, rotations=[-15, +15]):
   
    X_aug, y_aug = [], []

    for i in range(X.shape[0]):
        img = X[i].reshape(64, 64)
        label = y[i]

        # Image originale
        X_aug.append(img.flatten())
        y_aug.append(label)

        # Rotations
        for angle in rotations:
            rotated = rotate(img, angle, reshape=False, mode='nearest')
            X_aug.append(rotated.flatten())
            y_aug.append(label)


    X_aug, y_aug = np.array(X_aug), np.array(y_aug)
    return shuffle(X_aug, y_aug, random_state=42)
