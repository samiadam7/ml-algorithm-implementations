from base.base import BaseEstimator, BaseModel, TransformerMixin
import numpy as np

class UnivariateLDA(BaseEstimator, BaseModel):
    def __init__(self):
        super().__init__()
        
    def fit(self, X, y):
        self.is_fitted = True
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim != 1:
            raise ValueError("X must be 1-dimensional for univariate LDA.")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y arrays must be same length")
        
        best_error = np.inf
        
        for x_thresh in X:
            for y_ in [0,1]:    
                mistake = 0
                for xi, yi in zip(X, y):
                    if xi < x_thresh:
                        pred = 1- y_
                    else:
                        pred = y_
                        
                    if pred != yi:
                        mistake +=1
                        
                error = mistake / len(X)
                
                if error < best_error:
                    best_error = error
                    self.threshold = x_thresh
                    self.label_right = y_
    
    def predict(self, X):
        if X.ndim != 1:
            raise ValueError("X must be 1-dimensional for univariate LDA.")
        
        self._check_is_fitted()
        X = np.asarray(X)
        
        return np.where(X < self.threshold,1 - self.label_right, self.label_right)

class MulitvariateLDA(BaseEstimator, BaseModel, TransformerMixin):
    def __init__(self):
        super().__init__()
        
    def fit(self, X, y):
        self.is_fitted = True
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional for univariate LDA.")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y arrays must be same length")
        
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        
        means = np.zeros((len(self.classes), n_features))
        for i, k in enumerate(self.classes):
            X_k = X[y == k]
            means[i] = X_k.mean(axis= 0)
        
        self.class_means_ = means
        self.global_mean_ = X.mean(axis= 0)
        self.class_to_index_ = {label: idx for idx, label in enumerate(self.classes)}
        
        class_counts = np.array([len(X[y == k]) for k in self.classes])
        priors = class_counts / X.shape[0]
        self.priors_ = priors
        
        pooled_cov = np.zeros((n_features, n_features))
        
        for i, k in enumerate(self.classes):
            X_k = X[y == k]
            mean_k = self.class_means_[i]
            centered = X_k - mean_k
            
            pooled_cov += centered.T @ centered
            
        self.pooled_cov_ = pooled_cov / X.shape[0]
        
        evals_cov, evecs_cov = np.linalg.eigh(self.pooled_cov_)
        evals_cov = np.maximum(evals_cov, 1e-10)
        self.inv_cov_ = evecs_cov @ np.diag(1/evals_cov) @ evecs_cov.T
        
        between_class_cov = np.zeros((n_features, n_features))
        
        for i, k in enumerate(self.classes):
            mean_k = self.class_means_[i]
            centered = mean_k - self.global_mean_
            
            n_k = (y == k).sum() 
            between_class_cov += n_k * np.outer(centered, centered)

        self.between_class_cov_ = between_class_cov / X.shape[0]
        
        sqrt_inv_cov = evecs_cov @ np.diag(1 / np.sqrt(evals_cov)) @ evecs_cov.T
        A = sqrt_inv_cov @ self.between_class_cov_ @ sqrt_inv_cov
        
        evals, evecs = np.linalg.eigh(A)
        evals = np.maximum(evals, 0)
        
        sorted_idx = np.argsort(evals)[::-1]
        evecs = evecs[:, sorted_idx]
        
        K = len(self.classes)
        top_evecs = evecs[:,: K -1]
        
        self.W_ = top_evecs

    def transform(self, X):
        self._check_is_fitted()
        X = np.asarray(X)
        
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        
        return (X - self.global_mean_) @ self.W_
    
    def predict(self, X):
        self._check_is_fitted()
        
        X = np.asarray(X)
        
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        
        n_samples = X.shape[0]
        discriminant_scores = np.zeros((n_samples, len(self.classes)))
    
        for k in self.classes:
            k_idx = self.class_to_index_[k]
            k_mu = self.class_means_[k_idx]
            
            wk = self.inv_cov_ @ k_mu
            w0 = - (1/2) * k_mu.T @ self.inv_cov_ @ k_mu + np.log(self.priors_[k_idx])
            discriminant_scores[:, k_idx] = X @ wk + w0
            
        predicted_indices = np.argmax(discriminant_scores, axis=1)
        return self.classes[predicted_indices]

if __name__ == "__main__":
    lda = UnivariateLDA()
    X = np.array([1, 2, 3, 4, 5, 6])
    y = np.array([1, 1, 1, 0, 0, 0])
    lda.fit(X, y)
    print(lda.predict(X))
    
    lda = MulitvariateLDA()
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    y = np.array([1, 1, 1, 0, 0, 0])
    lda.fit(X, y)
    print(lda.predict(X))