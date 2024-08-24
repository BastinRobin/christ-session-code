import numpy as np 


class LogisticRegression:


    def __init__(self, lr=0.001, n_itern=100):
        self.lr = lr 
        self.n_itern = n_itern
        self.weights = None 
        self.bias = None 


    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))


    def fit(self, X, y):
        
        # Get the total features and total records
        n_samples, n_features = X.shape 

        # Initialize the weights and bias 
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training loops or epocs for each combination
        for _ in range(self.n_itern):

            # Create a linear model
            linear_model = np.dot(X, self.weights) + self.bias 
            y_pred = self._sigmoid(linear_model)
            
            # compute the weights and bias 
            dw = 1 / (n_samples) * np.dot(X.T, (y_pred - y))
            db = 1 / (n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db





    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)

        classes = [1 if i > 0.5 else 0 for i in y_pred]
        return np.array(classes)

