import numpy as np
from base.base import BaseEstimator, TransformerMixin

class PrincipalComponentAnalysis(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        
        if not isinstance(n_components, (int, float)):
            raise ValueError("n_components must be integer or floating point")
        
        self.n_components = n_components
        super().__init__()
        
    def fit(self, X):
        self._is_fitted = True
        
        X = np.asarray(X)
        self.n_samples_, self.n_features_ = X.shape
        
        self.mean_ = X.mean(axis= 0)
        X_centered = X - self.mean_
        cov = (1 / (self.n_samples_ - 1)) * X_centered.T @ X_centered
        
        e_vals, e_vecs = np.linalg.eigh(cov)
        idx = np.argsort(e_vals)[::-1]
        e_vals = e_vals[idx]
        e_vecs = e_vecs[:, idx]
        
        final_e_vals = e_vals[:self.n_components]
        final_e_vecs = e_vecs[:, :self.n_components]
        
        self._e_vals = final_e_vals
        self._e_vecs = final_e_vecs
        
        
        
        
    def transform(self, X):
        self._check_is_fitted()
        
        X = np.asarray(X)
        X_centered = X - self.mean_
        
        return X_centered @ self._e_vecs

        