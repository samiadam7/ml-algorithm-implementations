from base.base import BaseEstimator, BaseModel, TransformerMixin
import numpy as np

class UnivariateQDA(BaseEstimator, BaseModel, TransformerMixin):
    def __init__(self):
        super().__init__()
        
    def fit(self, X, y):
        self._is_fitted = True
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.dim != 1:
            raise ValueError("X must be 1-dimensional")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y arrays must be same length")
        
        
        
        
class MultivariateQDA(BaseEstimator, BaseModel, TransformerMixin):
    def __init__(self):
        super().__init__()