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

if __name__ == "__main__":
    print("=== PCA Tests ===")
    
    # Test 1: Basic functionality with integer n_components
    print("\nTest 1: Basic 2D to 1D reduction")
    X1 = np.array([[1, 1], [2, 2], [3, 3], [4, 0], [5, 1], [6, 2]])  # Points roughly along y=x
    pca = PrincipalComponentAnalysis(n_components=1)
    X1_transformed = pca.fit_transform(X1)
    print("Original shape:", X1.shape)
    print("Transformed shape:", X1_transformed.shape)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    
    # Test 2: Variance ratio threshold
    print("\nTest 2: Using variance ratio threshold")
    X2 = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 0.1]])  # Data with clear variance differences
    pca = PrincipalComponentAnalysis(n_components=0.95)  # Should keep 2 components
    X2_transformed = pca.fit_transform(X2)
    print("Original shape:", X2.shape)
    print("Transformed shape:", X2_transformed.shape)
    print("Explained variance ratios:", pca.explained_variance_ratio_)
    
    # Test 3: Edge case - requesting more components than features
    print("\nTest 3: Edge case - too many components")
    try:
        pca = PrincipalComponentAnalysis(n_components=5)
        pca.fit(X2)  # Should raise error (X2 only has 3 features)
    except ValueError as e:
        print("Correctly caught error:", str(e))
    
    # Test 4: Edge case - impossible variance threshold
    print("\nTest 4: Edge case - impossible variance threshold")
    pca = PrincipalComponentAnalysis(n_components=0.9999)
    X4 = np.random.randn(10, 3)  # Random data
    X4_transformed = pca.fit_transform(X4)
    print("Original shape:", X4.shape)
    print("Transformed shape:", X4_transformed.shape)
    print("Total variance explained:", sum(pca.explained_variance_ratio_))
    
    # Test 5: Method chaining
    print("\nTest 5: Method chaining")
    X5 = np.array([[1, 2], [3, 4], [5, 6]])
    pca = PrincipalComponentAnalysis(n_components=1)
    try:
        # These should work without errors
        X5_transformed = pca.fit(X5).transform(X5)
        print("Method chaining works!")
        X5_transformed_2 = pca.fit_transform(X5)
        print("fit_transform works!")
        assert np.allclose(X5_transformed, X5_transformed_2)
        print("Results are consistent!")
    except Exception as e:
        print("Error in method chaining:", str(e))

        