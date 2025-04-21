from base.base import BaseEstimator, BaseModel, TransformerMixin
import numpy as np

class UnivariateQDA(BaseEstimator, BaseModel, TransformerMixin):
    def __init__(self):
        super().__init__()
        
    def fit(self, X, y):
        self._is_fitted = True
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim != 1:
            raise ValueError("X must be 1-dimensional")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y arrays must be same length")
                
        self.classes = np.unique(y)
        
        self.means = np.zeros(len(self.classes))
        self.vars = np.zeros(len(self.classes))
        self.priors = np.zeros(len(self.classes))
        
        for i, k in enumerate(self.classes):
            X_k = X[y == k]
            
            self.means[i] = X_k.mean()
            self.vars[i] = X_k.var()
            self.priors[i] = len(X_k) / len(X)
            
        
    def predict(self, X):
        self._check_is_fitted()
        X = np.asarray(X)
        
        if X.ndim != 1:
            raise ValueError("X must be 1-dimensional")
        
        probs = np.zeros((len(X), len(self.classes)))
        
        for i, k in enumerate(self.classes):
            probs[:, i] = (-(1/2) * np.log(2 * np.pi * self.vars[i]  ** 2) - 
                        ((X - self.means[i])**2 / (2 * self.vars[i] ** 2)) + np.log(self.priors[i]))
            
        return self.classes[np.argmax(probs, axis= 1)]
            
        
        
class MultivariateQDA(BaseEstimator, BaseModel, TransformerMixin):
    def __init__(self):
        super().__init__()