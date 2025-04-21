import inspect

class BaseEstimator:
    def __init__(self):
        signature = inspect.signature(self.__init__)
        params = signature.parameters
        
        self._params = {}
        
        for name in params:
            if name == "self":
                continue
            else:
                self._params[name] = getattr(self, name)
                
    def get_params(self):
        return(self._params.copy())
        
    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)
            self._params[key] = val
            
    def __repr__(self):
        class_name = self.__class__.__name__
        param_str = ", ".join(f"{k}= {v!r}" for k, v in self._params.items())
        return f"{class_name}({param_str})"
    
class BaseModel:
    def __init__(self):
        self._is_fitted = False
    
    def fit(self, X, y):
        raise NotImplementedError("fit() must be implemented.")
    
    def predict(self, X):
        raise NotImplementedError("predict() must be implemented.")
    
    def _check_is_fitted(self):
        if not self._is_fitted:
            raise ValueError("Model instance has not been fit. Please call fit().")
        
class TransformerMixin:
    def fit_transform(self, X, y= None):
        if not hasattr(self, "transform"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not implement transform()."
                "fit_transform() requires transform to be defined."
            )
        self.fit(X, y)
        return self.transform(X)