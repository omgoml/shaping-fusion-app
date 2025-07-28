from NeuralNetwork import NeuralNetwork
import numpy as np
def main() -> None:
    random_data = np.random.rand(10)

    nn = NeuralNetwork(
        input_size = 10,
        hidden_layers = [8,8],
        output_size = 1,
        learning_rate = 0.01,
        activation = "relu",
        output_activation = "sigmoid",
        weight_initialization = "xavier"
    )
            
    print(random_data)
    print(nn.predict(random_data))
if __name__ == "__main__":
    main()
