import numpy as np
from numpy.typing import NDArray

class NeuralNetwork(object):
    def __init__(self,
        input_size:int,
        hidden_layers: list[int],
        output_size: int,
        learning_rate: float = 0.01,
        activation: str = "relu",
        output_activation: str = "sigmoid",
        weight_initialization: str = "xavier" 
        ) -> None:
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        """setting activate function for calculation""" 
        self.activation_function = self._get_activation_function(activation)
        self.activation_derivative_function = self._get_derivative_acivation_function(activation) 
        self.output_activation_function = self._get_activation_function(output_activation)
        self.output_activation_derivative = self._get_derivative_acivation_function(output_activation) 

        #structure of the entire model
        self.architectures: list[int] = [input_size] + hidden_layers + [output_size]

        #storing weight and bias for specific neuro
        self.weights: list[NDArray] = []
        self.biases: list[NDArray] = []
        
        self._initialize_parameter(weight_initialization)

    def _initialize_parameter(self, method: str) -> None:
        """initialize weight and bias for specific method"""
        for i in range(len(self.architectures) - 1):
            #the number of input datas (neuros in the input layers)
            fan_input = self.architectures[i]
            #the number of ouput date (neuros in the previous layer)
            fan_output = self.architectures[i + 1]

            if method == "xavier":
                #xavier initialization
                limit = np.sqrt(6.0 / (fan_input + fan_output))
                weight: NDArray = np.random.uniform(-limit, limit,(fan_output, fan_input))
            elif method == "he":
                #he initialization
                weight: NDArray = np.random.randn(fan_output,fan_input) * np.sqrt(2.0 / fan_input)
            else:
                #random weight 
                weight: NDArray = np.random.randn(fan_output,fan_input) * 0.01

            bias: NDArray = np.zeros(fan_output)

            self.weights.append(weight)
            self.biases.append(bias)

    def _softmax(self,x: NDArray):
        exp = np.exp(x - np.max(x,axis = 1, keepdims = True))
        return exp / np.sum(exp, axis = 1, keepdims = True)

    def _get_activation_function(self,type:str):
        function_type = {
            "relu": lambda x : np.maximum(0,x), 
            "sigmoid": lambda x: 1 / (1 + np.exp(-np.clip(x, -700,700))),
            "tanh": lambda x: np.tanh(x),
            "softmax": lambda x: self._softmax(x)
        }
        
        return function_type.get(type, function_type["relu"])
    
    def _get_derivative_acivation_function(self,type:str):
        derivative_type = {
            "relu": lambda x: (x > 0).astype(float),
            "sigmoid": lambda x: x * (1 - x),
            "tanh": lambda x: 1 - x**2, 
            "softmax": lambda x: np.ones_like(x)
        }
        
        return derivative_type.get(type, derivative_type["relu"])

    def forward(self,input_data:NDArray[np.float64]) -> NDArray:
        #storing input values from the previous layers  
        self.activation_values = [input_data]
        self.z_values = []
        
        """hidden layers calculation"""
        for i in range(len(self.weights) - 1):
            z_value = np.dot(self.weights[i], self.activation_values[i]) + self.biases[i]    
            self.z_values.append(z_value)
            
            activation = self.activation_function(z_value)
            self.activation_values.append(activation)
        
        #the resulte calculatied in the ouput_layer (last layers of the network)
        z_output = np.dot(self.weights[-1],self.activation_values[-1]) + self.biases[-1]
        self.z_values.append(z_output)

        output_value: NDArray = self.output_activation_function(z_output)
        self.activation_values.append(output_value)

        return output_value

    def predict(self,input_data):
        return self.forward(input_data)
