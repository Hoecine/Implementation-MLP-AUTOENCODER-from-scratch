import numpy as np
import pickle

from sklearn.preprocessing import OneHotEncoder

#j'ai utilisé le dataset qu'on a utilisé dans le TME3 de MAPSI
def load_usps_from_pkl(path="usps.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)

    X = data["X_train"]  # (n_samples, 256)
    y = data["Y_train"].reshape(-1, 1)  # (n_samples, 1)

    X = X.astype(np.float32) / 255.0

    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y)

    return X, y


#reshape pour utiliser le CNN

def load_usps_for_cnn(path="usps.pkl"):
    X, y = load_usps_from_pkl(path)
    X = X.reshape(-1, 1, 256)  # (n_samples, 1, 256) exactement comme en théorie
    return X, y