import numpy as np
from base.base import BaseEstimator, TransformerMixin, BaseModel
import warnings

class PrincipalComponentAnalysis(BaseEstimator, BaseModel, TransformerMixin):
    def __init__(self, n_components):
        
        if not isinstance(n_components, (int, float)):
            raise ValueError("n_components must be integer or floating point")
        
        if isinstance(n_components, int):
            self.n_components = n_components
        else:
            if 0 <= n_components <= 1:
                self.n_components = n_components
        super().__init__()
        
    def fit(self, X):
        self._is_fitted = True
        
        X = np.asarray(X)
        
        if X.ndim != 2:
            raise ValueError("X must be 2-Dimensional")
        self.n_samples_, self.n_features_ = X.shape
        
        if self.n_features_ < self.n_components:
            raise ValueError(f"n_components must be less than the number of features ({self.n_features_})")
        
        self.mean_ = X.mean(axis= 0)
        X_centered = X - self.mean_
        cov = (1 / (self.n_samples_ - 1)) * X_centered.T @ X_centered
        
        e_vals, e_vecs = np.linalg.eigh(cov)
        idx = np.argsort(e_vals)[::-1]
        e_vals = e_vals[idx]
        e_vecs = e_vecs[:, idx]
        
        denom = e_vals.sum()
        ratios = e_vals / denom
        
        self.explained_variance_ratio_ = ratios
        
        if isinstance(self.n_components, int):
            final_e_vals = e_vals[:self.n_components]
            final_e_vecs = e_vecs[:, :self.n_components]
            
        else:
            total_var = np.cumsum(self.explained_variance_ratio_)
            idx_arr = np.argwhere(total_var >= self.n_components)
            
            if len(idx_arr) == 0:
                warnings.warn(f"No number of components achieves {self.n_components:.2f} variance. Using all components.")
                final_e_vals = e_vals
                final_e_vecs = e_vecs
            
            else:
                idx = idx_arr[0][0]
                print(idx)
                final_e_vals = e_vals[:idx+1]
                final_e_vecs = e_vecs[:, :idx+1]
                
        self._e_vals = final_e_vals
        self._e_vecs = final_e_vecs
        
        return self
        
    def transform(self, X):
        self._check_is_fitted()
        
        X = np.asarray(X)
        X_centered = X - self.mean_
        
        return X_centered @ self._e_vecs

        