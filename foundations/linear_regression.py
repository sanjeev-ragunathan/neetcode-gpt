import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        # Y_hat = X @ np.transpose(weights)
        Y_hat = np.dot(X, weights)
        return np.round(Y_hat, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        # n = len(model_prediction)
        # squared_error = np.square(model_prediction - ground_truth)
        # mean_squared_error = np.sum(squared_error) / n
        mean_squared_error = np.mean((ground_truth - model_prediction) ** 2)
        return np.round(mean_squared_error, 5)
