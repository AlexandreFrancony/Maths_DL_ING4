import numpy as np

class SVM:
    def __init__(self, C=1.0, lr=0.01, max_iter=1000):
        """
        Initialize the SVM classifier with hyperparameters.

        Parameters:
        ----------
        C : float
            Regularization parameter for slack.
        lr : float
            Learning rate for gradient descent.
        max_iter : int
            Maximum number of iterations for training.
        """
        self.C = C
        self.lr = lr
        self.max_iter = max_iter
        self.w = None  # Weight vector
        self.b = None  # Bias term

    def _derivative_loss(self, xi, yi):
        """
        Compute the gradient of the loss function for a single sample.

        Parameters:
        ----------
        xi : ndarray
            Feature vector of a single sample.
        yi : int
            Label of the sample (+1 or -1).

        Returns:
        ----------
        d_w : ndarray
            Gradient w.r.t. the weights.
        d_b : float
            Gradient w.r.t. the bias.
        """
        condition = 1 - yi * (np.dot(self.w, xi) + self.b)
        if condition > 0:
            d_w = self.w - self.C * yi * xi
            d_b = -self.C * yi
        else:
            d_w = self.w
            d_b = 0
        return d_w, d_b

    def _loss(self, X, Y):
        """
        Compute the hinge loss for the dataset.

        Parameters:
        ----------
        X : ndarray
            Feature matrix of the dataset.
        Y : ndarray
            Labels for the dataset.

        Returns:
        ----------
        float
            Total loss value.
        """
        regularizer = 0.5 * np.dot(self.w, self.w)
        hinge_loss = np.maximum(0, 1 - Y * (np.dot(X, self.w) + self.b))
        error_term = self.C * np.sum(hinge_loss)
        return regularizer + error_term

    def fit(self, X, Y):
        """
        Train the SVM using stochastic gradient descent.

        Parameters:
        ----------
        X : ndarray
            Feature matrix of the training set.
        Y : ndarray
            Labels for the training set.
        """
        n_samples, n_features = X.shape
        self.w = np.ones(n_features)  # Initialize weights
        self.b = 1.0  # Initialize bias

        for _ in range(self.max_iter):
            for i in range(n_samples):
                d_w, d_b = self._derivative_loss(X[i], Y[i])
                self.w -= self.lr * d_w
                self.b -= self.lr * d_b

    def predict(self, X):
        """
        Predict labels for the input data.

        Parameters:
        ----------
        X : ndarray
            Feature matrix of the test set.

        Returns:
        ----------
        ndarray
            Predicted labels for the input data.
        """
        return np.sign(np.dot(X, self.w) + self.b)

    def plot_decision_boundary(self, X, Y):
        """
        Visualize the decision boundary (2D case only).

        Parameters:
        ----------
        X : ndarray
            Feature matrix (2D only).
        Y : ndarray
            Labels for the dataset.
        """
        if X.shape[1] != 2:
            raise ValueError("Plotting is only supported for 2D data.")

        import matplotlib.pyplot as plt

        # Scatter plot for data points
        plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color='blue', label='Class +1')
        plt.scatter(X[Y == -1][:, 0], X[Y == -1][:, 1], color='red', label='Class -1')

        # Plot the decision boundary
        x1 = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
        x2 = -(self.w[0] * x1 + self.b) / self.w[1]
        plt.plot(x1, x2, color='green', label='Decision Boundary')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.title('SVM Decision Boundary')
        plt.show()
