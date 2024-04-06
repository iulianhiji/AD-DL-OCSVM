import numpy as np
import scipy.io as sio
from utils import generate_dicts

if __name__ == "__main__":
    nu = {
        "satellite.mat": 0.317,
        "mnist.mat": 0.093,
        "shuttle.mat": 0.072,
        "pendigits.mat": 0.023,
        "speech.mat": 0.017,
        "cardio.mat": 0.096,
        "glass.mat": 0.042,
        "thyroid.mat": 0.025,
        "vowels.mat": 0.034,
        "wbc.mat": 0.056,
        "lympho.mat": 0.041,
    }

    for filename in nu.keys():
        y_all = sio.loadmat("../data/" + filename)
        y_all = y_all["X"]
        Y = y_all.T
        n_features = np.shape(Y)[0]  # signal dimension
        n_components = 2 * n_features  # number of atoms

        generate_dicts(20, Y.shape[0], n_components, filename)
        generate_dicts(20, Y.shape[1], n_components, filename, kernel=True)
