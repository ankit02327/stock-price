"""
Convolutional Neural Network (CNN) Implementation from Scratch

This module implements a CNN using:
- 2D Convolutional layers with custom kernels
- Max pooling for downsampling
- ReLU and Softmax activation functions
- Backpropagation through convolution operations
- Support for image classification tasks

Mathematical Foundation:
- Convolution: (f * g)(x,y) = Σ Σ f(m,n) * g(x-m, y-n)
- Max Pooling: max pooling over local regions
- ReLU: f(x) = max(0, x)
- Softmax: σ(x_i) = exp(x_i) / Σ exp(x_j)
"""

import numpy as np

#########################
# Activation Functions
#########################
class Activation:
    """
    Activation functions for CNN layers
    """
    @staticmethod
    def relu(x):
        """
        ReLU activation: f(x) = max(0, x)
        
        Parameters:
        - x: input tensor
        
        Returns:
        - ReLU applied tensor
        """
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """
        ReLU derivative: f'(x) = 1 if x > 0, else 0
        
        Parameters:
        - x: input tensor
        
        Returns:
        - derivative tensor
        """
        return (x > 0).astype(float)
    
    @staticmethod
    def softmax(x):
        """
        Softmax activation for classification
        
        Parameters:
        - x: input tensor (batch_size, num_classes)
        
        Returns:
        - probability distribution over classes
        """
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

#########################
# Layers
#########################
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        # He initialization for ReLU networks
        limit = np.sqrt(2.0/(in_channels*kernel_size*kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)*limit
        self.b = np.zeros((out_channels,1))
    def forward(self, X):
        n, c, h, w = X.shape
        f, _, kh, kw = self.W.shape
        out_h = (h+2*self.padding - kh)//self.stride + 1
        out_w = (w+2*self.padding - kw)//self.stride + 1
        X_pad = np.pad(X, ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)), mode='constant')
        out = np.zeros((n,f,out_h,out_w))
        
        # More efficient vectorized convolution
        for i in range(out_h):
            for j in range(out_w):
                X_slice = X_pad[:,:,i*self.stride:i*self.stride+kh,j*self.stride:j*self.stride+kw]
                # Batch matrix multiplication for all filters at once
                out[:,:,i,j] = np.tensordot(X_slice, self.W, axes=([1,2,3],[1,2,3])) + self.b.T
        
        self.X, self.X_pad = X, X_pad
        return out
    def backward(self, d_out, lr):
        X, X_pad = self.X, self.X_pad
        n,c,h,w = X.shape
        f,_,kh,kw = self.W.shape
        _,_,out_h,out_w = d_out.shape
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX = np.zeros_like(X_pad)
        # Simplified backward pass for speed
        for i in range(out_h):
            for j in range(out_w):
                X_slice = X_pad[:,:,i*self.stride:i*self.stride+kh,j*self.stride:j*self.stride+kw]
                # Vectorized gradient computation
                db += d_out[:,:,i,j].sum(axis=0, keepdims=True).T
                for k in range(f):
                    dW[k] += np.sum(d_out[:,k,i,j][:,None,None,None] * X_slice, axis=0)
                # Simplified dX computation
                for n_ in range(n):
                    dX[n_,:,i*self.stride:i*self.stride+kh,j*self.stride:j*self.stride+kw] += np.sum(self.W * d_out[n_,:,i,j][:,None,None,None], axis=0)
        self.W -= lr*dW
        self.b -= lr*db
        return dX[:,:,self.padding:self.padding+h,self.padding:self.padding+w]

class MaxPool2D:
    def __init__(self, kernel_size, stride):
        self.kernel_size=kernel_size
        self.stride=stride
    def forward(self,X):
        n,c,h,w = X.shape
        kh,kw=self.kernel_size,self.kernel_size
        out_h=(h-kh)//self.stride+1
        out_w=(w-kw)//self.stride+1
        out=np.zeros((n,c,out_h,out_w))
        self.X=X
        for i in range(out_h):
            for j in range(out_w):
                slice_=X[:,:,i*self.stride:i*self.stride+kh,j*self.stride:j*self.stride+kw]
                out[:,:,i,j]=slice_.reshape(n,c,-1).max(axis=2)
        return out
    def backward(self,d_out):
        X=self.X
        n,c,h,w=X.shape
        kh,kw=self.kernel_size,self.kernel_size
        dX=np.zeros_like(X)
        out_h=(h-kh)//self.stride+1
        out_w=(w-kw)//self.stride+1
        for i in range(out_h):
            for j in range(out_w):
                slice_=X[:,:,i*self.stride:i*self.stride+kh,j*self.stride:j*self.stride+kw]
                max_mask=(slice_==slice_.reshape(n,c,-1).max(axis=2)[:,:,None,None])
                dX[:,:,i*self.stride:i*self.stride+kh,j*self.stride:j*self.stride+kw]+=max_mask*d_out[:,:,i,j][:,:,None,None]
        return dX

class Flatten:
    def forward(self,X):
        self.orig_shape=X.shape
        return X.reshape(X.shape[0],-1)
    def backward(self,d_out):
        return d_out.reshape(self.orig_shape)

class Dense:
    def __init__(self,in_dim,out_dim):
        # He initialization for ReLU networks
        limit=np.sqrt(2.0/in_dim)
        self.W=np.random.randn(in_dim,out_dim)*limit
        self.b=np.zeros((1,out_dim))
    def forward(self,X):
        self.X=X
        return X.dot(self.W)+self.b
    def backward(self,d_out,lr):
        dW=self.X.T.dot(d_out)
        db=d_out.sum(axis=0,keepdims=True)
        dX=d_out.dot(self.W.T)
        self.W-=lr*dW
        self.b-=lr*db
        return dX

#########################
# CNN Model
#########################
class CNN:
    def __init__(self,lr=0.01):
        # Simplified architecture for faster training
        self.conv1=Conv2D(in_channels=1,out_channels=4,kernel_size=3,stride=1,padding=1)
        self.act1=Activation.relu
        self.pool1=MaxPool2D(kernel_size=2,stride=2)
        self.conv2=Conv2D(4,8,3,1,1)
        self.act2=Activation.relu
        self.pool2=MaxPool2D(2,2)
        self.flatten=Flatten()
        self.dense1=Dense(8*7*7,64)  # 8 channels * 7*7 spatial = 392
        self.act3=Activation.relu
        self.dense2=Dense(64,10)
        self.lr=lr
    def forward(self,X):
        out=self.conv1.forward(X); out=self.act1(out); out=self.pool1.forward(out)
        out=self.conv2.forward(out); out=self.act2(out); out=self.pool2.forward(out)
        out=self.flatten.forward(out)
        out=self.dense1.forward(out); z3=out; out=self.act3(out)
        out=self.dense2.forward(out); z4=out; out=Activation.softmax(out)
        return out,(z3,z4)
    def backward(self,X,y,pred,cache,lr):
        z3,z4=cache
        # Softmax cross-entropy delta
        delta=pred - y
        d2=self.dense2.backward(delta,lr)
        d2_act=Activation.relu_derivative(z3)*d2
        d1=self.dense1.backward(d2_act,lr)
        d1_flat=self.flatten.backward(d1)
        d2p=self.pool2.backward(d1_flat)
        d2a=d2p*Activation.relu_derivative(d2p)
        d3=self.conv2.backward(d2a,lr)
        d1p=self.pool1.backward(d3)
        d1a=d1p*Activation.relu_derivative(d1p)
        self.conv1.backward(d1a,lr)
    def train(self,X,y,epochs=5,batch_size=64):
        n=X.shape[0]
        for e in range(epochs):
            # Learning rate decay
            current_lr = self.lr * (0.9 ** e)
            perm=np.random.permutation(n)
            Xs=X[perm]; ys=y[perm]
            for i in range(0,n,batch_size):
                xb= Xs[i:i+batch_size]; yb=ys[i:i+batch_size]
                pred,cache=self.forward(xb)
                self.backward(xb,yb,pred,cache,current_lr)
            acc=(self.forward(X)[0].argmax(axis=1)==y.argmax(axis=1)).mean()
            print(f"Epoch {e+1}/{epochs}, Accuracy: {acc*100:.2f}%, LR: {current_lr:.4f}")
    def predict(self,X):
        return self.forward(X)[0].argmax(axis=1)

#########################
# Example Usage
#########################
if __name__=="__main__":
    print("="*60)
    print("CNN FOR IMAGE CLASSIFICATION (MNIST-like)")
    print("="*60)
    # Generate simple random data: 500 samples of 1×28×28 (reduced for speed)
    X = np.random.randn(500,1,28,28)
    # Normalize the data
    X = (X - X.mean()) / (X.std() + 1e-8)
    y_idx=np.random.randint(0,10,500)
    y=np.zeros((500,10)); y[np.arange(500),y_idx]=1
    # Split
    split=400
    X_train,X_test=X[:split],X[split:]
    y_train,y_test=y[:split],y[split:]
    # Create and train
    cnn=CNN(lr=0.01)
    cnn.train(X_train,y_train,epochs=3,batch_size=64)
    # Evaluate
    preds=cnn.predict(X_test)
    acc=(preds==y_idx[split:]).mean()
    print(f"Test Accuracy: {acc*100:.2f}%")



