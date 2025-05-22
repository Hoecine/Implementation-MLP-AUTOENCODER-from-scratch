import numpy as np
from projet_etu import Module 


# === Couches standard ===
"""
class Linear(Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._weights = np.random.randn(input_dim, output_dim) * 0.01
        self._bias = np.zeros((1, output_dim))
        self._grad_weights = np.zeros_like(self._weights)
        self._grad_bias = np.zeros_like(self._bias)

    def forward(self, X):
        self._input = X  # pour backprop
        return X @ self._weights + self._bias

    def backward_update_gradient(self, input, delta):
        self._grad_weights += input.T @ delta
        self._grad_bias += np.sum(delta, axis=0, keepdims=True)

    def backward_delta(self, input, delta):
        return delta @ self._weights.T

    def zero_grad(self):
        self._grad_weights[:] = 0
        self._grad_bias[:] = 0

    def update_parameters(self, gradient_step=1e-3):
        self._weights -= gradient_step * self._grad_weights
        self._bias -= gradient_step * self._grad_bias
"""
import numpy as np
from modules import Module  

class Linear(Module):
    def __init__(self, input_dim, output_dim, init_type=0, name="Linear"):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._name = name

        # Initialisation des poids
        self._parameters = self.init_weights(input_dim, output_dim, init_type)
        self._bias = self.init_weights(1, output_dim, init_type)

        self._gradient = np.zeros_like(self._parameters)
        self._gradient_bias = np.zeros_like(self._bias)

    def init_weights(self, n_in, n_out, init_type):
        np.random.seed(0)  # Pour résultats reproductibles
        if init_type == 'xavier' or init_type == 1:
            return np.random.normal(0, 1, (n_in, n_out)) * np.sqrt(2 / (n_in + n_out))
        elif init_type == 'he' or init_type == 2:
            return np.random.normal(0, 1, (n_in, n_out)) * np.sqrt(2 / n_in)
        else:  # aléatoire standard
            return np.random.randn(n_in, n_out)

    def forward(self, X):
        self.input = X
        return X @ self._parameters + self._bias

    def backward_update_gradient(self, input, delta):
        self._gradient += input.T @ delta
        self._gradient_bias += np.sum(delta, axis=0, keepdims=True)

    def backward_delta(self, input, delta):
        return delta @ self._parameters.T

    def zero_grad(self):
        self._gradient.fill(0)
        self._gradient_bias.fill(0)

    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step * self._gradient
        self._bias -= gradient_step * self._gradient_bias

class Tanh(Module):
    def forward(self, X):
        self.output = np.tanh(X)
        return self.output

    def backward_update_gradient(self, input, delta):
        pass  

    def backward_delta(self, input, delta):
        return (1 - self.output ** 2) * delta

    def update_parameters(self, gradient_step):
        pass
    
class Sigmoide(Module):
    def forward(self, X):
        self.output = 1 / (1 + np.exp(-X))  # On sauvegarde la sortie pour le backward
        return self.output

    def backward_update_gradient(self, input, delta):
        pass  

    def backward_delta(self, input, delta):
        return self.output * (1 - self.output) * delta

    def update_parameters(self, gradient_step):
        pass  
class ReLU(Module):
    def forward(self, X):
        self.input = X  # ok
        return np.maximum(0, X)

    def backward_update_gradient(self, input, delta):
        pass  # pas de params à mettre à jour

    def backward_delta(self, input, delta):
        return delta * (self.input > 0).astype(float)

    def zero_grad(self):
        pass

    def update_parameters(self, gradient_step):
        pass

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)

    def forward(self, X):
        self.inputs = [X]
        out = X
        for module in self.modules:
            out = module.forward(out)
            self.inputs.append(out)
        return out

    def backward_update_gradient(self, input, delta):
        # On va utiliser les inputs stockés pour chaque couche
        for i in reversed(range(len(self.modules))):
            module = self.modules[i]
            current_input = self.inputs[i]
            current_output = self.inputs[i + 1]  # sortie après cette couche

            if hasattr(module, 'backward_update_gradient'):
                module.backward_update_gradient(current_input, delta)

            if hasattr(module, 'backward_delta'):
                delta = module.backward_delta(current_output, delta)

    def update_parameters(self, gradient_step):
        for module in self.modules:
            module.update_parameters(gradient_step)

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()



class LogSoftmax(Module):
    def forward(self, X):
        X_stable = X - np.max(X, axis=1, keepdims=True)  # stabilité numérique
        exp_X = np.exp(X_stable)
        sum_exp = np.sum(exp_X, axis=1, keepdims=True)
        self.output = X_stable - np.log(sum_exp)  # log(softmax)
        return self.output #fonction qui calcule la softmax de vecteurs de sorties
        #c'est pour que les sorties du réseaux soient des probas pour chaque classe

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        softmax = np.exp(self.output)
        return softmax - (-delta) # dérivée simplifiée de log(softmax)

    def update_parameters(self, gradient_step):
        pass


# === CNN ===

class Conv1D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        # 1 filtre par out_channel, appliqué à chaque in_channel
        self._parameters = np.random.randn(out_channels, in_channels, kernel_size) * 0.01
        self._gradient = np.zeros_like(self._parameters)

    def forward(self, X):
        """
        X : (batch_size, in_channels, input_length)
        Retourne Y : (batch_size, out_channels, output_length)
        """
        self.X = X 
        batch_size, in_channels, input_length = X.shape
        output_length = (input_length - self.kernel_size) // self.stride + 1
        Y = np.zeros((batch_size, self.out_channels, output_length))
        
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for t in range(output_length): 
                    for ic in range(in_channels):
                        window = X[b, ic, t*self.stride : t*self.stride + self.kernel_size]
                        Y[b, oc, t] += np.sum(window * self._parameters[oc, ic])
        return Y
    
    def backward_update_gradient(self, input, delta):
        """
        input : même shape que self.X (batch_size, in_channels, input_length)
        delta : dérivée de la loss par rapport à Y (même shape que Y)
        """
        batch_size, _, input_length = input.shape
        output_length = delta.shape[2]
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for t in range(output_length):
                    for ic in range(self.in_channels):
                        window = input[b, ic, t*self.stride : t*self.stride + self.kernel_size]
                        self._gradient[oc, ic] += delta[b, oc, t] * window
    
    def backward_delta(self, input, delta):
        """
        Retourne l'erreur à propager à la couche précédente
        (même shape que l'entrée : batch, in_channels, input_length)
        """
        batch_size, _, input_length = input.shape
        output_length = delta.shape[2]
        d_input = np.zeros_like(input)
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for t in range(output_length):
                    for ic in range(self.in_channels):
                        d_input[b, ic, t*self.stride : t*self.stride + self.kernel_size] += (delta[b, oc, t] * self._parameters[oc, ic])

        return d_input
    def zero_grad(self):
        self._gradient = np.zeros_like(self._parameters)

    def update_parameters(self, gradient_step):
        self._parameters -= gradient_step * self._gradient



class MaxPool1D(Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, X):
        """
        X : (batch_size, in_channels, input_length)
        """
        self.X = X  # pour le backward

        batch_size, in_channels, input_length = X.shape
        output_length = (input_length - self.kernel_size) // self.stride + 1
        Y = np.zeros((batch_size, in_channels, output_length))

        self.max_indices = np.zeros_like(Y, dtype=int)

        for b in range(batch_size):
            for c in range(in_channels):
                for t in range(output_length):
                    window = X[b, c, t*self.stride : t*self.stride + self.kernel_size]
                    Y[b, c, t] = np.max(window)
                    self.max_indices[b, c, t] = np.argmax(window) + t*self.stride  # on stocke la position du max

        return Y
    
class Flatten(Module):
    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)  # (n, c*l)

    def backward_delta(self, input, delta):
        return delta.reshape(self.input_shape)

    def update_parameters(self, step):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def zero_grad(self):
        pass

