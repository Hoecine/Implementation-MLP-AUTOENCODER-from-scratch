import numpy as np

class Optim:
    def __init__(self, net, loss, eps=1e-3):
        self.net = net        # un Sequential
        self.loss = loss      # une instance de Loss
        self.eps = eps        # pas de gradient

    def step(self, batch_x, batch_y):
        yhat = self.net.forward(batch_x)
        loss_value = self.loss.forward(batch_y, yhat)

        delta = self.loss.backward(batch_y, yhat)
        self.net.zero_grad()
        self.net.backward_update_gradient(batch_x, delta)
        self.net.update_parameters(self.eps)

        return float(np.mean(loss_value))


def SGD(network, loss, X, y, eps=1e-3, batch_size=32, n_epochs=100, verbose=True):
    """
    Entraîne un réseau avec descente de gradient stochastique.

    - network : OBJET de Sequential
    - loss : objet de loss de Loss
    - X, y : données d'entrée et labels
    - eps : pas de gradient
    - batch_size : taille d'un mini-batch
    - n_epochs : nombre de passes sur les données
    """
    opt = Optim(network, loss, eps)
    n = X.shape[0]
    history = []
    for epoch in range(n_epochs):
        # Mélange des données
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        epoch_loss = 0

        for i in range(0, n, batch_size):
            xb = X_shuffled[i:i+batch_size]
            yb = y_shuffled[i:i+batch_size]
            loss_vals = opt.step(xb, yb) #vecteur des pertes (1 perte par exemple)
            epoch_loss += loss_vals #perte moyenne par batch
        
        epoch_loss /= (n // batch_size)
        history.append(epoch_loss)
        
        if verbose:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")
            
    return history