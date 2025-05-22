import numpy as np
from projet_etu import Loss

class MSELoss(Loss):
    def forward(self, y, yhat):
        # Renvoie un vecteur de pertes pour chaque élément du batch
        return 0.5 * np.sum((y - yhat) ** 2, axis=1) #pour chaque exemple sa perte (donc on aura un vecteur de pertes du batch complet)

    def backward(self, y, yhat):
        return yhat - y
    
class NLLLoss(Loss):
    def forward(self, y, yhat):
        # y : one-hot (obligatoirement)
        # yhat : log-probas
        return -np.sum(y * yhat, axis=1)  # NLL = -log(p_true_class)

    def backward(self, y, yhat):
        # Gradient = -y, car on suppose que yhat = log(softmax)
        return -y
class BCELoss(Loss):
    def forward(self, y, yhat):
        # Évite les log(0)
        yhat = np.clip(yhat, 1e-9, 1 - 1e-9)
        return - (y * np.log(yhat) + (1 - y) * np.log(1 - yhat)).mean()

    def backward(self, y, yhat):
        yhat = np.clip(yhat, 1e-9, 1 - 1e-9)
        return (yhat - y)/y.shape[0]
