import numpy as np
from numpy.typing import NDArray

class Solution:

    def sigmoid(self, z: float) -> float:
        return 1 / (1 + np.exp(-z))
    
    def relu(self, z: float) -> float:
        return max(0, z)
    
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)

        weighted_sum = x @ w
        z = weighted_sum + b
        if activation == "sigmoid":
            result = self.sigmoid(z)
        elif activation == "relu":
            result = self.relu(z)
        
        return round(float(result), 5)
