"""
Autoencoder for Dimensionality Reduction Implementation from Scratch

This module implements a neural network autoencoder using:
- Feedforward neural networks with multiple activation functions
- Backpropagation algorithm for training
- Encoder-decoder architecture for dimensionality reduction
- Reconstruction loss optimization

Mathematical Foundation:
- Forward Pass: h = f(Wx + b), x' = f(W'h + b')
- Loss Function: L = ||x - x'||² (Mean Squared Error)
- Backpropagation: ∂L/∂W = ∂L/∂x' * ∂x'/∂W
- Gradient Descent: W = W - α * ∂L/∂W
"""

import numpy as np

#########################
# Activation Functions
#########################

class Activation:
    """Collection of activation functions and their derivatives"""
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid: σ(x) = 1 / (1 + e^(-x))"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of sigmoid"""
        s = Activation.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def relu(x):
        """ReLU: max(0, x)"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    @staticmethod
    def tanh(x):
        """Tanh activation"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        """Derivative of tanh"""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def linear(x):
        """Linear activation"""
        return x
    
    @staticmethod
    def linear_derivative(x):
        """Derivative of linear"""
        return np.ones_like(x)

#########################
# Autoencoder Class
#########################

class Autoencoder:
    """
    Autoencoder for Dimensionality Reduction
    
    Architecture:
    Input -> Encoder -> Latent (compressed) -> Decoder -> Reconstruction
    
    Goal: Learn compressed representation while minimizing reconstruction error
    """
    
    def __init__(self, input_dim, encoding_dim, activation='sigmoid', learning_rate=0.01, random_seed=42):
        """
        Initialize Autoencoder
        
        Parameters:
        - input_dim: dimension of input data
        - encoding_dim: dimension of compressed representation (bottleneck)
        - activation: activation function ('sigmoid', 'relu', 'tanh')
        - learning_rate: learning rate for gradient descent
        - random_seed: for reproducibility
        """
        np.random.seed(random_seed)
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
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
        else:
            self.activation = Activation.sigmoid
            self.activation_derivative = Activation.sigmoid_derivative
        
        # Initialize encoder weights
        # Encoder: input_dim -> encoding_dim
        limit_encoder = np.sqrt(6.0 / (input_dim + encoding_dim))
        self.W_encoder = np.random.uniform(-limit_encoder, limit_encoder, 
                                           (input_dim, encoding_dim))
        self.b_encoder = np.zeros((1, encoding_dim))
        
        # Initialize decoder weights
        # Decoder: encoding_dim -> input_dim
        limit_decoder = np.sqrt(6.0 / (encoding_dim + input_dim))
        self.W_decoder = np.random.uniform(-limit_decoder, limit_decoder,
                                           (encoding_dim, input_dim))
        self.b_decoder = np.zeros((1, input_dim))
        
        # Training history
        self.loss_history = []
    
    def encode(self, X):
        """
        Encode input to latent representation
        
        Formula: z = σ(X @ W_encoder + b_encoder)
        """
        z_linear = np.dot(X, self.W_encoder) + self.b_encoder
        z = self.activation(z_linear)
        return z, z_linear
    
    def decode(self, z):
        """
        Decode latent representation to reconstruction
        
        Formula: X_reconstructed = σ(z @ W_decoder + b_decoder)
        """
        X_linear = np.dot(z, self.W_decoder) + self.b_decoder
        X_reconstructed = self.activation(X_linear)
        return X_reconstructed, X_linear
    
    def forward(self, X):
        """
        Forward pass through autoencoder
        
        Returns:
        - X_reconstructed: reconstructed input
        - z: encoded (compressed) representation
        - cache: intermediate values for backpropagation
        """
        # Encoding
        z, z_linear = self.encode(X)
        
        # Decoding
        X_reconstructed, X_linear = self.decode(z)
        
        # Cache for backprop
        cache = {
            'X': X,
            'z': z,
            'z_linear': z_linear,
            'X_reconstructed': X_reconstructed,
            'X_linear': X_linear
        }
        
        return X_reconstructed, z, cache
    
    def compute_loss(self, X, X_reconstructed):
        """
        Compute reconstruction loss (MSE)
        
        Loss = (1/2m) * ||X - X_reconstructed||²
        """
        m = X.shape[0]
        loss = np.sum((X - X_reconstructed) ** 2) / (2 * m)
        return loss
    
    def backward(self, cache):
        """
        Backward propagation to compute gradients
        
        Returns:
        - gradients: dictionary of weight and bias gradients
        """
        X = cache['X']
        z = cache['z']
        z_linear = cache['z_linear']
        X_reconstructed = cache['X_reconstructed']
        X_linear = cache['X_linear']
        
        m = X.shape[0]
        
        # Output layer error (reconstruction)
        dX_reconstructed = (X_reconstructed - X) / m
        
        # Gradient through decoder activation
        dX_linear = dX_reconstructed * self.activation_derivative(X_linear)
        
        # Decoder gradients
        dW_decoder = np.dot(z.T, dX_linear)
        db_decoder = np.sum(dX_linear, axis=0, keepdims=True)
        
        # Backpropagate to encoder
        dz = np.dot(dX_linear, self.W_decoder.T)
        
        # Gradient through encoder activation
        dz_linear = dz * self.activation_derivative(z_linear)
        
        # Encoder gradients
        dW_encoder = np.dot(X.T, dz_linear)
        db_encoder = np.sum(dz_linear, axis=0, keepdims=True)
        
        gradients = {
            'dW_encoder': dW_encoder,
            'db_encoder': db_encoder,
            'dW_decoder': dW_decoder,
            'db_decoder': db_decoder
        }
        
        return gradients
    
    def update_parameters(self, gradients):
        """Update weights and biases using gradient descent"""
        self.W_encoder -= self.learning_rate * gradients['dW_encoder']
        self.b_encoder -= self.learning_rate * gradients['db_encoder']
        self.W_decoder -= self.learning_rate * gradients['dW_decoder']
        self.b_decoder -= self.learning_rate * gradients['db_decoder']
    
    def fit(self, X, epochs=1000, batch_size=32, verbose=True):
        """
        Train the autoencoder
        
        Parameters:
        - X: input data (m x input_dim)
        - epochs: number of training iterations
        - batch_size: size of mini-batches
        - verbose: whether to print progress
        """
        m = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            
            # Mini-batch gradient descent
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                
                # Forward pass
                X_reconstructed, z, cache = self.forward(X_batch)
                
                # Backward pass
                gradients = self.backward(cache)
                
                # Update parameters
                self.update_parameters(gradients)
            
            # Compute loss on full dataset
            X_reconstructed, _, _ = self.forward(X)
            loss = self.compute_loss(X, X_reconstructed)
            self.loss_history.append(loss)
            
            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
    
    def transform(self, X):
        """
        Encode input to compressed representation
        
        Parameters:
        - X: input data
        
        Returns:
        - z: encoded representation (m x encoding_dim)
        """
        z, _ = self.encode(X)
        return z
    
    def reconstruct(self, X):
        """
        Reconstruct input from autoencoder
        
        Parameters:
        - X: input data
        
        Returns:
        - X_reconstructed: reconstructed data
        """
        X_reconstructed, _, _ = self.forward(X)
        return X_reconstructed
    
    def fit_transform(self, X, epochs=1000, batch_size=32, verbose=True):
        """
        Fit autoencoder and return compressed representation
        """
        self.fit(X, epochs, batch_size, verbose)
        return self.transform(X)

#########################
# Example Usage
#########################

if __name__ == "__main__":
    print("=" * 60)
    print("AUTOENCODER FOR DIMENSIONALITY REDUCTION")
    print("=" * 60)
    
    # Example 1: Compress 10D data to 3D
    print("\\n--- Example 1: 10D -> 3D Compression ---")
    
    np.random.seed(42)
    
    # Generate synthetic 10D data
    n_samples = 500
    input_dim = 10
    
    # Data with some correlation structure
    X = np.random.randn(n_samples, input_dim)
    # Add some structure
    X[:, 1] = X[:, 0] + np.random.randn(n_samples) * 0.5
    X[:, 2] = X[:, 0] - X[:, 1] + np.random.randn(n_samples) * 0.5
    
    # Normalize data to [0, 1]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    
    # Create autoencoder: 10D -> 3D -> 10D
    autoencoder = Autoencoder(
        input_dim=10,
        encoding_dim=3,
        activation='sigmoid',
        learning_rate=0.1,
        random_seed=42
    )
    
    # Train
    print("\\nTraining autoencoder...")
    autoencoder.fit(X, epochs=1000, batch_size=64, verbose=False)
    
    # Get compressed representation
    X_compressed = autoencoder.transform(X)
    
    # Reconstruct
    X_reconstructed = autoencoder.reconstruct(X)
    
    # Compute reconstruction error
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    
    print(f"\\nOriginal dimension: {X.shape[1]}")
    print(f"Compressed dimension: {X_compressed.shape[1]}")
    print(f"Compression ratio: {input_dim / 3:.2f}x")
    print(f"Reconstruction MSE: {reconstruction_error:.6f}")
    print(f"Final training loss: {autoencoder.loss_history[-1]:.6f}")
    
    # Example 2: Visualization with 2D bottleneck
    print("\\n--- Example 2: 5D -> 2D for Visualization ---")
    
    # Generate data with 3 clusters
    np.random.seed(123)
    cluster1 = np.random.randn(150, 5) + np.array([5, 5, 5, 5, 5])
    cluster2 = np.random.randn(150, 5) + np.array([-5, -5, -5, -5, -5])
    cluster3 = np.random.randn(150, 5) + np.array([5, -5, 0, 5, -5])
    
    X_clustered = np.vstack([cluster1, cluster2, cluster3])
    labels = np.array([0]*150 + [1]*150 + [2]*150)
    
    # Normalize
    X_clustered = (X_clustered - X_clustered.min(axis=0)) / (X_clustered.max(axis=0) - X_clustered.min(axis=0) + 1e-8)
    
    # Create 2D autoencoder for visualization
    autoencoder_2d = Autoencoder(
        input_dim=5,
        encoding_dim=2,
        activation='sigmoid',
        learning_rate=0.5,
        random_seed=42
    )
    
    print("\\nTraining 2D autoencoder...")
    autoencoder_2d.fit(X_clustered, epochs=1000, batch_size=64, verbose=False)
    
    # Get 2D representation
    X_2d = autoencoder_2d.transform(X_clustered)
    
    print(f"\\nCompressed to 2D for visualization")
    print(f"Reconstruction loss: {autoencoder_2d.loss_history[-1]:.6f}")
    
    # Calculate compression statistics
    print(f"\nCompression Statistics:")
    print(f"10D -> 3D: {input_dim/3:.2f}x compression")
    print(f"5D -> 2D: {5/2:.2f}x compression")
    
    # Calculate variance explained by latent dimensions
    latent_var_2d = np.var(X_2d, axis=0)
    total_var_2d = np.sum(latent_var_2d)
    explained_var_2d = latent_var_2d / total_var_2d * 100
    print(f"Variance explained by 2D latent dimensions: {explained_var_2d}")
    
    # Calculate reconstruction error statistics
    errors_3d = np.mean((X - X_reconstructed) ** 2, axis=1)
    errors_2d = np.mean((X_clustered - autoencoder_2d.reconstruct(X_clustered)) ** 2, axis=1)
    print(f"Mean reconstruction error (3D): {np.mean(errors_3d):.6f}")
    print(f"Mean reconstruction error (2D): {np.mean(errors_2d):.6f}")
    
    print("\\n" + "=" * 60)
    print("AUTOENCODER IMPLEMENTATION COMPLETE!")
    print("=" * 60)