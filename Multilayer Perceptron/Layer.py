import numpy as np
from typing import Optional

class Layer:

    def __init__(
        self, 
        input_size: np.ndarray, 
        output_size: np.ndarray, 
        activation: str,
        a_linear: Optional[float] = 1.0,
        b_linear: Optional[float] = 0,
      ):
        self.weights = np.random.rand(output_size, input_size)

        self.stimulus = None
        self.local_field = None
        self.output = None

        self.local_gradient = None
        self.weights_grad = None

        self.activation = activation
        self.a_linear = a_linear
        self.b_linear = b_linear

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.stimulus = inputs
        self.local_field = np.matmul(self.weights, self.stimulus)
        self.output = self.phi(self.local_field)

        return self.output

    def backward(self, previous_gradient: np.ndarray) -> np.ndarray:
        self.local_gradient = np.multiply(
            previous_gradient, self.compute_derivative(self.local_field)
        )
        self.weights_grad = np.matmul(self.local_gradient, self.stimulus.T)
        return np.matmul(self.weights.T, self.local_gradient)

    def compute_derivative(self, v:np.ndarray) -> np.ndarray:
        if(self.activation == "Linear"):
          return np.zeros(v.shape)+1
        elif(self.activation == "Sigmoid"):
            return self.phi(v)*\
            (1-self.phi(v))
        elif(self.activation == "Tanh"):
            return 1 - self.phi(v)**2

    def update(self, learning_rate: float) -> None:
        self.weights = self.weights + learning_rate * self.weights_grad

    def phi(self, v: np.ndarray) -> np.ndarray:
        if(self.activation == "Linear"):
          return self.a_linear*v + self.b_linear
        elif(self.activation == "Sigmoid"):
            return 1/(1+np.exp(-v))
        elif(self.activation == "Tanh"):
            return np.tanh(v)
