"""
Feedforward Neural Network Implementation from Scratch

This module implements a multi-layer perceptron (MLP) using:
- Multiple activation functions (sigmoid, ReLU, tanh, linear)
- Backpropagation algorithm for training
- Gradient descent optimization
- Support for classification and regression tasks

Mathematical Foundation:
- Forward Pass: a^(l) = f(W^(l) * a^(l-1) + b^(l))
- Loss Function: L = (1/2m) * Σ||y - ŷ||² (MSE) or Cross-Entropy
- Backpropagation: ∂L/∂W^(l) = ∂L/∂a^(l) * ∂a^(l)/∂W^(l)
- Gradient Descent: W^(l) = W^(l) - α * ∂L/∂W^(l)
"""

import numpy as np

#########################
# Activation Functions
#########################

class Activation:
    """Collection of activation functions and their derivatives"""
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid activation: σ(x) = 1 / (1 + e^(-x))"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of sigmoid: σ'(x) = σ(x) * (1 - σ(x))"""
        s = Activation.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def relu(x):
        """ReLU activation: max(0, x)"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """Derivative of ReLU: 1 if x > 0, else 0"""
        return (x > 0).astype(float)
    
    @staticmethod
    def tanh(x):
        """Tanh activation: (e^x - e^(-x)) / (e^x + e^(-x))"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        """Derivative of tanh: 1 - tanh^2(x)"""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def linear(x):
        """Linear activation: f(x) = x"""
        return x
    
    @staticmethod
    def linear_derivative(x):
        """Derivative of linear: 1"""
        return np.ones_like(x)

#########################
# Neural Network Class
#########################

class NeuralNetwork:
    """
    Feedforward Neural Network with Backpropagation
    
    Features:
    - Configurable architecture (arbitrary number of layers)
    - Multiple activation functions
    - Multiple loss functions
    - Mini-batch gradient descent
    - Training history tracking
    """
    
    def __init__(self, layer_sizes, activation='sigmoid', learning_rate=0.01, random_seed=42):
        """
        Initialize neural network
        
        Parameters:
        - layer_sizes: list of integers, e.g., [2, 4, 3, 1] means:
            - 2 input features
            - Hidden layer 1: 4 neurons
            - Hidden layer 2: 3 neurons
            - Output layer: 1 neuron
        - activation: 'sigmoid', 'relu', 'tanh', or 'linear'
        - learning_rate: learning rate for gradient descent
        - random_seed: for reproducibility
        """
        np.random.seed(random_seed)
        
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        
        # Set activation function
        if activation == 'sigmoid':
            self.activation = Activation.sigmoid
            self.activation_derivative = Activation.sigmoid_derivative
        elif activation == 'relu':
            self.activation = Activation.relu
            self.activation_derivative = Activation.relu_derivative
        elif activation == 'tanh':
            self.activation = Activation.tanh
            self.activation_derivative = Activation.tanh_derivative
        elif activation == 'linear':
            self.activation = Activation.linear
            self.activation_derivative = Activation.linear_derivative
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            # Xavier/He initialization
            limit = np.sqrt(2.0 / layer_sizes[i])
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * limit
            b = np.zeros((1, layer_sizes[i + 1]))
            
            self.weights.append(w)
            self.biases.append(b)
        
        # Training history
        self.loss_history = []
    
    def forward_propagation(self, X):
        """
        Forward pass through the network
        
        Returns:
        - activations: list of activations for each layer (including input)
        - z_values: list of pre-activation values
        """
        activations = [X]
        z_values = []
        
        for i in range(self.num_layers - 1):
            # Linear combination: z = X @ W + b
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # Activation: a = σ(z)
            a = self.activation(z)
            activations.append(a)
        
        return activations, z_values
    
    def compute_loss(self, y_true, y_pred, loss_type='mse'):
        """
        Compute loss
        
        Parameters:
        - loss_type: 'mse' (mean squared error) or 'binary_crossentropy'
        """
        m = y_true.shape[0]
        
        if loss_type == 'mse':
            loss = np.mean((y_true - y_pred) ** 2)
        elif loss_type == 'binary_crossentropy':
            # Clip predictions to prevent log(0)
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        return loss
    
    def backward_propagation(self, X, y, activations, z_values, loss_type='mse'):
        """
        Backward pass to compute gradients
        
        Returns:
        - weight_gradients: list of weight gradients
        - bias_gradients: list of bias gradients
        """
        m = X.shape[0]
        weight_gradients = []
        bias_gradients = []
        
        # Output layer error
        if loss_type == 'mse':
            # For MSE: dL/da = 2(a - y)
            delta = (activations[-1] - y) * self.activation_derivative(z_values[-1])
        elif loss_type == 'binary_crossentropy':
            # For binary cross-entropy with sigmoid: dL/dz = a - y
            delta = activations[-1] - y
        else:
            delta = (activations[-1] - y) * self.activation_derivative(z_values[-1])
        
        # Backpropagate through layers
        for i in range(self.num_layers - 2, -1, -1):
            # Gradients for current layer
            dW = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)
            
            # Propagate error to previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(z_values[i - 1])
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients, bias_gradients):
        """Update weights and biases using gradient descent"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def train(self, X, y, epochs=1000, batch_size=32, loss_type='mse', verbose=True):
        """
        Train the neural network
        
        Parameters:
        - X: input data (m x n)
        - y: target values (m x output_size)
        - epochs: number of training iterations
        - batch_size: size of mini-batches
        - loss_type: 'mse' or 'binary_crossentropy'
        - verbose: whether to print progress
        """
        m = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch gradient descent
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Forward pass
                activations, z_values = self.forward_propagation(X_batch)
                
                # Backward pass
                weight_gradients, bias_gradients = self.backward_propagation(
                    X_batch, y_batch, activations, z_values, loss_type
                )
                
                # Update parameters
                self.update_parameters(weight_gradients, bias_gradients)
            
            # Compute loss on full dataset
            activations, _ = self.forward_propagation(X)
            loss = self.compute_loss(y, activations[-1], loss_type)
            self.loss_history.append(loss)
            
            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
    
    def predict(self, X):
        """Make predictions"""
        activations, _ = self.forward_propagation(X)
        return activations[-1]
    
    def evaluate(self, X, y, loss_type='mse'):
        """Evaluate model on test data"""
        predictions = self.predict(X)
        loss = self.compute_loss(y, predictions, loss_type)
        
        # For binary classification
        if loss_type == 'binary_crossentropy':
            y_pred_class = (predictions > 0.5).astype(int)
            accuracy = np.mean(y_pred_class == y)
            return loss, accuracy
        
        return loss

#########################
# Example Usage
#########################

if __name__ == "__main__":
    print("=" * 60)
    print("FEEDFORWARD NEURAL NETWORK - FROM SCRATCH")
    print("=" * 60)
    
    # Example 1: XOR Problem (classic neural network example)
    print("\\n--- Example 1: XOR Problem ---")
    
    # XOR dataset
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])
    
    # Create neural network: 2 inputs -> 4 hidden -> 1 output
    nn_xor = NeuralNetwork(
        layer_sizes=[2, 4, 1],
        activation='sigmoid',
        learning_rate=0.5,
        random_seed=42
    )
    
    # Train
    nn_xor.train(X_xor, y_xor, epochs=5000, batch_size=4, 
                 loss_type='binary_crossentropy', verbose=False)
    
    # Test
    predictions = nn_xor.predict(X_xor)
    print("\\nXOR Predictions:")
    for i in range(len(X_xor)):
        print(f"Input: {X_xor[i]}, Predicted: {predictions[i][0]:.4f}, True: {y_xor[i][0]}")
    
    # Example 2: Simple Regression
    print("\\n--- Example 2: Simple Regression ---")
    
    # Generate synthetic data: y = 2x + 3
    np.random.seed(42)
    X_reg = np.random.rand(100, 1) * 10
    y_reg = 2 * X_reg + 3 + np.random.randn(100, 1) * 0.5
    
    # Create neural network: 1 input -> 5 hidden -> 1 output
    nn_reg = NeuralNetwork(
        layer_sizes=[1, 5, 1],
        activation='relu',
        learning_rate=0.01,
        random_seed=42
    )
    
    # Train
    nn_reg.train(X_reg, y_reg, epochs=1000, batch_size=32, 
                 loss_type='mse', verbose=False)
    
    # Test on a few samples
    X_test = np.array([[2.0], [5.0], [8.0]])
    predictions = nn_reg.predict(X_test)
    print("\\nRegression Predictions:")
    for i in range(len(X_test)):
        true_val = 2 * X_test[i][0] + 3
        print(f"Input: {X_test[i][0]:.2f}, Predicted: {predictions[i][0]:.2f}, True: {true_val:.2f}")
    
    # Display training statistics
    print(f"\nXOR Training Statistics:")
    print(f"Final loss: {nn_xor.loss_history[-1]:.6f}")
    print(f"Training epochs: {len(nn_xor.loss_history)}")
    
    print(f"\nRegression Training Statistics:")
    print(f"Final loss: {nn_reg.loss_history[-1]:.6f}")
    print(f"Training epochs: {len(nn_reg.loss_history)}")
    
    # Calculate accuracy for XOR
    xor_accuracy = np.mean((predictions > 0.5) == y_xor)
    print(f"XOR Accuracy: {xor_accuracy*100:.2f}%")
    
    # Calculate MSE for regression
    reg_mse = np.mean((predictions - (2 * X_test + 3))**2)
    print(f"Regression MSE: {reg_mse:.6f}")
    
    print("\\n" + "=" * 60)
    print("NEURAL NETWORK IMPLEMENTATION COMPLETE!")
    print("=" * 60)