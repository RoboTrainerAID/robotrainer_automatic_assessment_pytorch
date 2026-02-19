from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from .sklearn_base import SklearnBaseModel

class LinearReg(SklearnBaseModel):
    model_name = "LinearRegression"
    
    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)
        model = LinearRegression()
        # Ensure multi-output support
        self.model = MultiOutputRegressor(model)

    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            "correlation_threshold": trial.suggest_float("correlation_threshold", 0.4, 0.8, step=0.01)}

    @staticmethod
    def get_default_parameters():
        return {}


class ElasticNetReg(SklearnBaseModel):
    model_name = "ElasticNet"
    
    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)
        
        alpha = hyperparams.get("alpha", 0.1)
        l1_ratio = hyperparams.get("l1_ratio", 0.5)
        
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000, random_state=42)
        self.model = MultiOutputRegressor(model)

    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "correlation_threshold": trial.suggest_float("correlation_threshold", 0.4, 0.8, step=0.01)
        }

    @staticmethod
    def get_default_parameters():
        return {"alpha": 0.1, "l1_ratio": 0.5}


class SVRReg(SklearnBaseModel):
    model_name = "SVR"
    
    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)
        
        C = hyperparams.get("C", 1.0)
        epsilon = hyperparams.get("epsilon", 0.1)
        kernel = hyperparams.get("kernel", "rbf")
        
        model = SVR(C=C, epsilon=epsilon, kernel=kernel)
        self.model = MultiOutputRegressor(model)

    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]), #, "poly"
            "correlation_threshold": trial.suggest_float("correlation_threshold", 0.4, 0.8, step=0.01)
        }

    @staticmethod
    def get_default_parameters():
        # General defaults for SVR
        return {"C": 1.0, "epsilon": 0.1, "kernel": "rbf"}


class RandomForestReg(SklearnBaseModel):
    model_name = "RandomForest"
    
    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)
        
        n_estimators = hyperparams.get("n_estimators", 100)
        max_depth = hyperparams.get("max_depth", None)
        
        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1 # Use all cores
        )
        self.model = MultiOutputRegressor(model)

    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20]),
            "correlation_threshold": trial.suggest_float("correlation_threshold", 0.4, 0.8, step=0.01)
        }

    @staticmethod
    def get_default_parameters():
        return {"n_estimators": 100, "max_depth": None}

class SGDReg(SklearnBaseModel):
    model_name = "SGDRegressor"
    
    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)
        
        penalty = hyperparams.get("penalty", "l2")
        alpha = hyperparams.get("alpha", 1e-4) # Regularization term
        
        model = SGDRegressor(penalty=penalty, alpha=alpha, max_iter=5000, tol=1e-3, random_state=42)
        self.model = MultiOutputRegressor(model)

    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            "penalty": trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "correlation_threshold": trial.suggest_float("correlation_threshold", 0.4, 0.8, step=0.01)
        }

    @staticmethod
    def get_default_parameters():
        return {"penalty": "l2", "alpha": 1e-4}
