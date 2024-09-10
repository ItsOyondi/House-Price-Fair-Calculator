
import numpy as np
import matplotlib.pyplot as plt
import util
## KNN Classifier - chosen because of its higher accuracy
class KNNClassifier():
  #defining training function
  def fit(self, X, y):
    self.X = X
    self.y = y

  #defining prediction function
  def predict(self, X, K, epsilon = 1e-3):
    N = len(X)
    y_hat = np.zeros(N)
    X = np.array(X)
    self.X = np.array(self.X)
     # get squared values outside the loop to reduce computation time
    X_square_sum = np.sum(X**2, axis=1)
    train_square_sum = np.sum(self.X**2, axis=1)
    dist2 = -2 * (X @ self.X.T) + X_square_sum[:, None] + train_square_sum

    for i in range(N):
        idxt = np.argsort(dist2[i])[:K]
        gamma_k = 1 / (np.sqrt(dist2[i, idxt] + epsilon))
        y_hat[i] = np.bincount(self.y[idxt], weights=gamma_k).argmax()

    return y_hat.astype(int)

  def accuracy(self, y, y_pred):
    return np.mean(y==y_pred)

## Multivariate Regression model
class MVLinearRegression():
  def fit(self, X, y, eta=1e-3, epochs=1e3, show_curve=False):
    epochs = int(epochs)
    N, D = X.shape
    Y = y
    #Begin optimizations
    self.w = np.random.randn(D)
    self.j = np.zeros(epochs)

    #stochastic Gradient Descent
    for epoch in range(epochs):
      y_hat = self.predict(X)
      self.j[epoch] = util.OLS(X,y, y_hat)
      #weight update rule
      self.w -= eta*(1/N)*(X.T@(y_hat-Y))
    if show_curve:
      plt.figure(figsize=(5,4))
      plt.plot(self.j)
      plt.xlabel('Epochs')
      plt.ylabel('$\mathcal{j}$')
      plt.title('Training Curve')
      plt.show()
  def predict(self, X):
    return X@self.w
# MV Ridge regression (L2)
class RidgeRegression():
    def fit(self, X, y, eta=1e-3, epochs=1e3, lambda_=1, show_curve=False):
        epochs = int(epochs)
        N, D = X.shape
        Y = y
        self.w = np.random.randn(D)
        self.j = np.zeros(epochs)

        # Stochastic Gradient Descent
        for epoch in range(epochs):
            y_hat = self.predict(X)
            # Calculate the cost function with # L2 regularization
            self.j[epoch] = util.OLS(X, y, y_hat) + (lambda_/2) * np.sum(self.w**2)

            # calculate gradient against the weights
            grad_w = (1/N) * (X.T @ (y_hat - Y))
            self.w -= eta * (grad_w + lambda_ * self.w)

        if show_curve:
            plt.figure(figsize=(5,4))
            plt.plot(self.j)
            plt.xlabel('Epochs')
            plt.ylabel('$\mathcal{j}$')
            plt.title('Training Curve')
            plt.show()

    def predict(self, X):
        return X @ self.w

# KNN Regressor
class KNNRegressor():
  def fit(self, X, y):
    self.X = X
    self.y = y
  def predict(self, X, k, epsilon=1e-3):
    N = len(X)
    y_hat = np.zeros(N)
    for i in range(N):
      dist2 = np.sum((self.X-X[i])**2, axis=1)
      idx = np.argsort(dist2)[:k]
      gamma_k = np.exp(-dist2[idx])/(np.exp(dist2[idx]).sum() + epsilon)
      y_hat[i] = gamma_k @ self.y[idx]

    return y_hat