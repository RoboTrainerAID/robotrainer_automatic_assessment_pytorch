import warnings

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
        fit_intercept = hyperparams.get("fit_intercept", True)
        model = LinearRegression(fit_intercept=fit_intercept)
        self.model = MultiOutputRegressor(model)

    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "n_path_features": trial.suggest_int("n_path_features", 20, 70, step=5),
        }

    @staticmethod
    def get_default_parameters():
        return {"fit_intercept": False, "n_path_features": 30}


class ElasticNetReg(SklearnBaseModel):
    model_name = "ElasticNet"
    
    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)
        
        alpha = hyperparams.get("alpha", 0.1)
        l1_ratio = hyperparams.get("l1_ratio", 0.5)
        fit_intercept = hyperparams.get("fit_intercept", True)
        selection = hyperparams.get("selection", "cyclic")
        
        model = ElasticNet(
            alpha=alpha, 
            l1_ratio=l1_ratio, 
            fit_intercept=fit_intercept,
            selection=selection,
            max_iter=20000, 
            random_state=42
        )
        self.model = MultiOutputRegressor(model)

    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "selection": trial.suggest_categorical("selection", ["cyclic", "random"]),
            "n_path_features": trial.suggest_int("n_path_features", 20, 70, step=5),
        }

    @staticmethod
    def get_default_parameters():
        return {"alpha": 0.2, "l1_ratio": 0.4, "fit_intercept": False, "selection": "cyclic", "n_path_features": 30}


class SVRReg(SklearnBaseModel):
    model_name = "SVR"
    
    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)
        
        C = hyperparams.get("C", 1.0)
        epsilon = hyperparams.get("epsilon", 0.1)
        kernel = hyperparams.get("kernel", "rbf")
        gamma = hyperparams.get("gamma", "scale")
        degree = hyperparams.get("degree", 3)
        shrinking = hyperparams.get("shrinking", True)
        
        model = SVR(
            C=C, 
            epsilon=epsilon, 
            kernel=kernel, 
            gamma=gamma,
            degree=degree,
            shrinking=shrinking
        )
        self.model = MultiOutputRegressor(model)

    @staticmethod
    def get_hyperparameter_space(trial):
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        
        params = {
            "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
            "kernel": kernel,
            "shrinking": trial.suggest_categorical("shrinking", [True, False]),
            "n_path_features": trial.suggest_int("n_path_features", 20, 70, step=5),
        }
        
        # Conditional parameters based on kernel
        if kernel in ["rbf", "poly"]:
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
        
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 5)
        
        return params

    @staticmethod
    def get_default_parameters():
        return {"C": 47.94113098310296, "epsilon": 0.2795553581365559, "kernel": "linear", "gamma": "scale", "shrinking": True, "n_path_features": 30}


class RandomForestReg(SklearnBaseModel):
    model_name = "RandomForest"
    
    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)
        
        n_estimators = hyperparams.get("n_estimators", 100)
        max_depth = hyperparams.get("max_depth", None)
        min_samples_split = hyperparams.get("min_samples_split", 2)
        min_samples_leaf = hyperparams.get("min_samples_leaf", 1)
        max_features = hyperparams.get("max_features", 1.0)
        bootstrap = hyperparams.get("bootstrap", True)
        
        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=42,
            n_jobs=-1
        )
        self.model = MultiOutputRegressor(model)

    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0, step=0.1),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "n_path_features": trial.suggest_int("n_path_features", 20, 70, step=5),
        }

    @staticmethod
    def get_default_parameters():
        return {
            "n_estimators": 500, "max_depth": 20, "min_samples_split": 2,
            "min_samples_leaf": 1, "max_features": 0.5, "bootstrap": True, 
            "n_path_features": 60
        }


class AutoSklearnReg(SklearnBaseModel):
    """AutoML baseline using auto-sklearn's AutoSklearnRegressor.
    
    Auto-sklearn performs its own internal model selection and hyperparameter
    optimization. The Optuna layer in the pipeline therefore exposes only
    high-level resource-budget parameters.
    
    Install:  pip install auto-sklearn
    Note: auto-sklearn requires Linux and Python <=3.10.
    """
    model_name = "AutoSklearn"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)
        try:
            from autosklearn.regression import AutoSklearnRegressor
        except ImportError:
            raise ImportError(
                "auto-sklearn is not installed. "
                "Install it with: pip install auto-sklearn\n"
                "Note: auto-sklearn requires Linux and Python <=3.10."
            )

        time_left = hyperparams.get("time_left_for_this_task", 120)
        per_run_time = hyperparams.get("per_run_time_limit", 30)
        ensemble_size = hyperparams.get("ensemble_size", 1)
        initial_configs = hyperparams.get("initial_configurations_via_metalearning", 25)
        memory_limit = hyperparams.get("memory_limit", 4096)  # MB

        self.model = AutoSklearnRegressor(
            time_left_for_this_task=int(time_left),
            per_run_time_limit=int(per_run_time),
            ensemble_class="default" if ensemble_size > 0 else None,
            initial_configurations_via_metalearning=int(initial_configs),
            memory_limit=int(memory_limit),
            n_jobs=-1,
            seed=42,
        )

    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            "time_left_for_this_task": trial.suggest_categorical(
                "time_left_for_this_task", [60, 120, 300, 600]
            ),
            "per_run_time_limit": trial.suggest_categorical(
                "per_run_time_limit", [10, 30, 60]
            ),
            "ensemble_size": trial.suggest_int("ensemble_size", 0, 50, step=5),
            "initial_configurations_via_metalearning": trial.suggest_int(
                "initial_configurations_via_metalearning", 0, 25, step=5
            ),
            "n_path_features": trial.suggest_int("n_path_features", 20, 70, step=5),
        }

    @staticmethod
    def get_default_parameters():
        return {
            "time_left_for_this_task": 120,
            "per_run_time_limit": 30,
            "ensemble_size": 1,
            "initial_configurations_via_metalearning": 25,
            "memory_limit": 4096,
            "n_path_features": 50,
        }

    @classmethod
    def print_summary(cls, X, y):
        print(f"\n--- AutoML Model Summary: {cls.__name__} ---")
        print("auto-sklearn AutoML regressor (internal model selection + HPO).")
        print("Input: Flattened (Path Features + User Features)")
        print(f"Output Targets: {y.shape[1]}")
        print("---------------------------------------")


class TabPFNReg(SklearnBaseModel):
    """Tabular foundation model baseline using TabPFN.
    
    TabPFN is a pre-trained transformer that performs well on small-to-medium
    tabular datasets (up to ~10k rows).  It follows the sklearn estimator API.
    
    Since TabPFN is a pre-trained foundation model, it has very few tunable
    parameters (only n_estimators and feature selection).  Hyperparameter
    tuning is lightweight — consider using 'default' mode or very few trials.
    
    Install:  pip install tabpfn
    Requires: Python >=3.9, GPU strongly recommended.
    """
    model_name = "TabPFN"

    @staticmethod
    def _resolve_device(requested: str = "auto") -> str:
        """Pick the best available device. 'auto' -> cuda if available."""
        import torch
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return requested

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)
        try:
            from tabpfn import TabPFNRegressor
        except ImportError:
            raise ImportError(
                "TabPFN is not installed.\n"
                "Requires Python >=3.9. GPU strongly recommended.\n"
                "Here are the steps to set up TabPFN:\n"
                "pip install tabpfn --no-deps && pip install tabpfn-common-utils einops eval-type-backport huggingface-hub pydantic pydantic-settings\n"
                "export <hugging_face_token.txt>\n"
            )

        device = self._resolve_device(hyperparams.get("device", "auto"))
        n_estimators = hyperparams.get("n_estimators", 4)

        # Build a single-output TabPFN regressor, then wrap for multi-output
        base_regressor = TabPFNRegressor(
            device=device,
            n_estimators=n_estimators,
            random_state=42,
        )
        self.model = MultiOutputRegressor(base_regressor)

    @staticmethod
    def get_hyperparameter_space(trial):
        # TabPFN is a pre-trained foundation model — very few knobs to turn.
        # n_estimators: ensemble size (higher = better accuracy, slower inference)
        # n_path_features: pipeline-level feature selection (shared across all sklearn models)
        return {
            "n_estimators": trial.suggest_int("n_estimators", 4, 16),
            "n_path_features": trial.suggest_int("n_path_features", 20, 70, step=5),
        }

    @staticmethod
    def get_default_parameters():
        return {
            "n_estimators": 4,
            "device": "auto",  # auto-detects GPU; set "cpu" to force CPU
            "n_path_features": 50,
        }

    @classmethod
    def print_summary(cls, X, y):
        print(f"\n--- Foundation Model Summary: {cls.__name__} ---")
        print("TabPFN pre-trained tabular transformer (regressor).")
        print("Input: Flattened (Path Features + User Features)")
        print(f"Output Targets: {y.shape[1]}")
        print("---------------------------------------")


class SGDReg(SklearnBaseModel):
    model_name = "SGDRegressor"
    
    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)
        
        penalty = hyperparams.get("penalty", "l2")
        alpha = hyperparams.get("alpha", 1e-4)
        l1_ratio = hyperparams.get("l1_ratio", 0.15)
        learning_rate = hyperparams.get("learning_rate", "invscaling")
        eta0 = hyperparams.get("eta0", 0.01)
        power_t = hyperparams.get("power_t", 0.25)
        epsilon_sgd = hyperparams.get("epsilon_sgd", 0.1)
        fit_intercept = hyperparams.get("fit_intercept", True)
        
        model = SGDRegressor(
            penalty=penalty, 
            alpha=alpha,
            l1_ratio=l1_ratio,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            epsilon=epsilon_sgd,
            fit_intercept=fit_intercept,
            max_iter=5000, 
            tol=1e-3, 
            random_state=42
        )
        self.model = MultiOutputRegressor(model)

    @staticmethod
    def get_hyperparameter_space(trial):
        penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
        learning_rate = trial.suggest_categorical("learning_rate", ["constant", "optimal", "invscaling", "adaptive"])
        
        params = {
            "penalty": penalty,
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "learning_rate": learning_rate,
            "eta0": trial.suggest_float("eta0", 1e-4, 1.0, log=True),
            "epsilon_sgd": trial.suggest_float("epsilon_sgd", 1e-3, 1.0, log=True),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "n_path_features": trial.suggest_int("n_path_features", 20, 70, step=5),
        }
        
        # Conditional: l1_ratio only relevant for elasticnet
        if penalty == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
        
        # Conditional: power_t only relevant for invscaling
        if learning_rate == "invscaling":
            params["power_t"] = trial.suggest_float("power_t", 0.1, 0.5)
        
        return params

    @staticmethod
    def get_default_parameters():
        return {
            "penalty": "l2", "alpha": 1e-4, "l1_ratio": 0.15,
            "learning_rate": "invscaling", "eta0": 0.01, "power_t": 0.25,
            "epsilon_sgd": 0.1, "fit_intercept": True, "n_path_features": 50
        }
