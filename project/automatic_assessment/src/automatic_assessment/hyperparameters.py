import pandas as pd
from skopt.space import Real, Categorical, Integer

class HyperparameterManager:
    """
    Manages default hyperparameters and search spaces for Bayesian Optimization.
    Uses a nested dictionary structure: Model -> Defaults/Dataset -> Params.
    """
    
    def __init__(self, model_type: str, dataset_type: str, n_features: int, n_samples: int):
        """
        Args:
            model_type: 'svr' or 'rf'.
            dataset_type: 'user', 'path', or time interval string.
            n_features: Number of input features.
            n_samples: Number of available training samples.
        """
        self.model_type = model_type
        self.dataset_type = dataset_type
        self.n_features = n_features
        
        # Calculate safe upper bound for PCA
        self.safe_pca_limit = min(n_features, n_samples - 1)
        if self.safe_pca_limit < 1:
            self.safe_pca_limit = 1
            
        # Initialize the Configuration Dictionary
        self._configs = self._build_configs()

    def _build_configs(self):
        """
        Builds the nested dictionary structure.
        Structure:
        {
            'svr': {
                'defaults': { 'param_name': {'best': val, 'space': skopt_obj}, ... },
                'user': { 'param_name': {'best': val, 'space': skopt_obj} }, # Overrides
                'path': { ... }
            },
            'rf': { ... }
        }
        """
        configs = {
            'svr': {
                'defaults': {
                    'pca__n_components': {
                        # Number of PCA components'
                        'best': min(5, self.safe_pca_limit), 
                        'space': Integer(2, max(2, self.safe_pca_limit)),
                    },
                    'regressor__estimator__C': {
                        # Regularization parameter'
                        'best': 1.0, 
                        'space': Real(0.01, 100, prior='log-uniform'),
                    },
                    'regressor__estimator__epsilon': {
                        # Epsilon tube width'
                        'best': 0.1, 
                        'space': Real(0.01, 1.0, prior='log-uniform'),
                    },
                    'regressor__estimator__gamma': {
                        # Kernel coefficient'
                        'best': 'scale', 
                        'space': Categorical(['scale', 'auto']),
                    },
                    'regressor__estimator__kernel': {
                        # Kernel type'
                        'best': 'rbf', 
                        'space': Categorical(['rbf']),
                    }
                },
                'user': {
                    # Specific overrides for User-based dataset (Small N)
                    'pca__n_components': {
                        'best': min(15, self.safe_pca_limit),
                        'space': Integer(2, max(2, self.safe_pca_limit))
                    }
                },
                'path': {
                    # Specific overrides for Path-based dataset (Large N)
                    'pca__n_components': {'best': 27},
                    'regressor__estimator__C': {'best': 0.37}
                }
            },
            'rf': {
                'defaults': {
                    'pca__n_components': {
                        'best': min(10, self.safe_pca_limit),
                        'desc': 'Number of PCA components'
                    },
                    'regressor__estimator__n_estimators': {
                        'best': 100, 
                        'space': Integer(50, 500),
                        'desc': 'Number of trees'
                    },
                    'regressor__estimator__max_depth': {
                        'best': None, 
                        'space': Integer(3, 20),
                        'desc': 'Max tree depth'
                    },
                    'regressor__estimator__min_samples_split': {
                        'best': 2, 
                        'space': Integer(2, 10),
                        'desc': 'Min samples to split'
                    }
                },
                'user': {}, # No specific overrides yet
                'path': {}  # No specific overrides yet
            }
        }
        return configs

    def _get_merged_config(self) -> dict:
        """Helper to merge defaults with dataset-specific overrides for the current model."""
        if self.model_type not in self._configs:
            print(f"Warning: Model type '{self.model_type}' not found.")
            return {}

        model_config = self._configs[self.model_type]
        
        # Start with defaults
        merged = model_config.get('defaults', {}).copy()
        
        # Apply overrides if they exist for this dataset type
        if self.dataset_type in model_config:
            overrides = model_config[self.dataset_type]
            for param, values in overrides.items():
                # Update existing param or add new one
                if param in merged:
                    merged[param] = merged[param].copy() # Shallow copy to avoid mutating defaults
                    merged[param].update(values)
                else:
                    merged[param] = values
                    
        return merged

    def get_best_params(self) -> dict:
        """Retrieves best parameters for the current dataset interval and model."""
        merged_config = self._get_merged_config()
        best_params = {}
        
        for param, details in merged_config.items():
            if 'best' in details:
                best_params[param] = details['best']
        
        # Safety Check: Clip PCA if it exceeds current limit
        if 'pca__n_components' in best_params and best_params['pca__n_components'] > self.safe_pca_limit:
            print(f"Config Warning: Clipping n_components from {best_params['pca__n_components']} to {self.safe_pca_limit}")
            best_params['pca__n_components'] = self.safe_pca_limit
            
        return best_params

    def get_search_space(self) -> dict:
        """Retrieves search space for the current dataset interval and model."""
        merged_config = self._get_merged_config()
        search_space = {}
        
        for param, details in merged_config.items():
            if 'space' in details:
                search_space[param] = details['space']
                
        return search_space

    def export_configurations(self, output_path: str = "/data/hyperparameter_config.csv"):
        """Exports the resolved configurations for the current dataset to CSV."""
        rows = []
        
        # Export for all supported models for the CURRENT dataset type
        for model_type in self._configs.keys():
            # Temporarily switch context to export all models
            original_model = self.model_type
            self.model_type = model_type
            merged_config = self._get_merged_config()
            self.model_type = original_model # Restore
            
            for param, details in merged_config.items():
                rows.append({
                    'Dataset Type': self.dataset_type,
                    'Model Type': model_type,
                    'Parameter': param,
                    'Best Value': str(details.get('best', 'N/A')),
                    'Search Space': str(details.get('space', 'N/A')),
                    'Description': details.get('desc', '')
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Hyperparameter configurations exported to {output_path}")