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
    
    def fit(self, X, y):
        self._is_fitted = True
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional for Multivariate QDA")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X ({X.shape[0]}) and y ({y.shape[0]}) must be the same length")
        
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.n_classes_ = len(self.classes)
        self.class_to_index_ = {label: idx for idx, label in enumerate(self.classes)}
        
        means = np.zeros((self.n_classes_, n_features))
        covs = np.zeros((self.n_classes_, n_features, n_features))
        
        for i, k in enumerate(self.classes):
            X_k = X[y == k]
            mean_k = np.mean(X_k, axis= 0)
            
            centered = X_k - mean_k
            cov_k = (centered.T @ centered) / len(X_k)
            cov_k += np.eye(n_features) * 1e-6
            
            means[i] = mean_k
            covs[i] = cov_k
            
        self.class_means_ = means
        self.covariances_ = covs
        
        class_counts = np.array([len(X[y == k]) for k in self.classes])
        priors = class_counts / X.shape[0]
        self.priors_ = priors

    def predict(self, X):
        self._check_is_fitted()
        
        X = np.asarray(X)
        
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        
        probs = np.zeros((len(X), self.n_classes_))
        for i, k in enumerate(self.classes):
            cov_k = self.covariances_[i]
            mean_k = self.class_means_[i]
            
            t1 = -0.5 * np.log(np.linalg.det(cov_k))
            
            diff = X - mean_k
            inv_cov_k = np.linalg.inv(cov_k)
            t2 = -0.5 * np.sum((diff @ inv_cov_k) * diff, axis=1)
            
            probs[:, i] = t1 + t2 + np.log(self.priors_[i])
            
        return self.classes[np.argmax(probs, axis= 1)]
        
if __name__ == "__main__":
    print("=== UnivariateQDA Tests ===")
    
    # Test 1: Basic binary classification
    print("\nTest 1: Basic binary classification")
    uni_qda = UnivariateQDA()
    X1 = np.array([1, 2, 3, 4, 5, 6])
    y1 = np.array([1, 1, 1, 0, 0, 0])
    uni_qda.fit(X1, y1)
    print("Training predictions:", uni_qda.predict(X1))
    print("Single point prediction:", uni_qda.predict(np.array([2.5])))

    # Test 2: Classes with different variances
    print("\nTest 2: Different variances")
    X2 = np.array([1, 1.1, 5, 5.1, 5.2, 10])
    y2 = np.array([0, 0, 1, 1, 1, 0])  # Class 1 has smaller variance
    uni_qda.fit(X2, y2)
    print("Training predictions:", uni_qda.predict(X2))
    
    print("\n=== MultivariateQDA Tests ===")
    
    # Test 1: Simple 2D binary classification
    print("\nTest 1: Simple 2D binary classification")
    multi_qda = MultivariateQDA()
    X1 = np.array([[1, 2], [2, 3], [3, 4],    # class 0
                   [6, 7], [7, 8], [8, 9]])    # class 1
    y1 = np.array([0, 0, 0, 1, 1, 1])
    multi_qda.fit(X1, y1)
    print("Training predictions:", multi_qda.predict(X1))
    print("New points prediction:", multi_qda.predict(np.array([[2, 3], [7, 8]])))
    
    # Test 2: Three classes with different covariances
    print("\nTest 2: Three classes with different covariances")
    X2 = np.array([[0, 0], [1, 0], [0, 1],     # tight cluster (class 0)
                   [5, 5], [7, 5], [5, 7],      # spread out cluster (class 1)
                   [2, 4], [3, 4], [2.5, 3]])   # medium cluster (class 2)
    y2 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    multi_qda.fit(X2, y2)
    print("Training predictions:", multi_qda.predict(X2))
    print("New points prediction:", 
          multi_qda.predict(np.array([[0.5, 0.5],  # should be class 0
                                    [6, 6],        # should be class 1
                                    [2.5, 3.5]]))) # should be class 2
    
    # Test 3: Unbalanced classes
    print("\nTest 3: Unbalanced classes")
    X3 = np.array([[1, 1], [1.1, 1], [0.9, 1],     # many class 0
                   [1.2, 0.8], [0.8, 1.2], [1, 0.9],
                   [5, 5], [5.1, 5.1]])             # few class 1
    y3 = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    multi_qda.fit(X3, y3)
    print("Training predictions:", multi_qda.predict(X3))
    print("New points prediction:", 
          multi_qda.predict(np.array([[1, 1],   # should be class 0
                                    [5, 5]])))  # should be class 1