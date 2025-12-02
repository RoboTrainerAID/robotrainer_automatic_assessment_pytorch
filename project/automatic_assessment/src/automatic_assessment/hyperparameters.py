import pandas as pd
from skopt.space import Real, Categorical, Integer

class HyperparameterManager:
    """
    Manages default hyperparameters and search spaces for Bayesian Optimization.
    Uses a nested dictionary structure: Model -> Defaults/Dataset -> Params.
    """
    
    def __init__(self, model_type: str, dataset_type: str):
        """
        Args:
            model_type: 'svr' or 'rf'.
            dataset_type: 'user', 'path', or time interval string.
        """
        self.model_type = model_type
        self.dataset_type = dataset_type
            
        # Initialize the Configuration Dictionary
        self._configs = self._build_configs()

    def _build_configs(self) -> dict:
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
                        'best': 15, 
                        'space': Integer(5, 76),
                        'desc': 'Number of PCA components'
                    },
                    'regressor__estimator__C': {
                        'best': 1.0, 
                        'space': Real(0.01, 100, prior='log-uniform'),
                        'desc': 'Regularization parameter'
                    },
                    'regressor__estimator__epsilon': {
                        'best': 0.1, 
                        'space': Real(0.01, 1.0, prior='log-uniform'),
                        'desc': 'Epsilon tube width'
                    },
                    'regressor__estimator__gamma': {
                        'best': 'scale', 
                        'space': Categorical(['scale', 'auto']),
                        'desc': 'Kernel coefficient'
                    },
                    'regressor__estimator__kernel': {
                        'best': 'rbf', 
                        'space': Categorical(['rbf']),
                        'desc': 'Kernel type'
                    }
                },
                'user': {
                    # Specific overrides for User-based dataset (Small N)
                    'pca__n_components': {
                        'best': 15,
                        'space': Integer(2, 24)
                    }
                },
                'path': {
                    # Specific overrides for Path-based dataset (Large N)
                    'pca__n_components': {'best': 38},
                    'regressor__estimator__C': {'best': 0.28}
                }
            },
            'rf': {
                'defaults': {
                    # 1. SPEED: Cap this at 150. 
                    # 100 is usually the sweet spot for small data.
                    'regressor__n_estimators': {
                        'best': 79,
                        'space': Integer(50, 200), 
                        'desc': 'Number of trees'
                    },
                    
                    # 2. OVERFITTING: Cap this at 10. 
                    # Going deeper than 10 for 560 samples is just memorizing noise.
                    'regressor__max_depth': {
                        'best': 7,
                        'space': Integer(3, 10),
                        'desc': 'Max tree depth'
                    },
                    
                    # 3. SMOOTHING: Add min_samples_leaf.
                    # Forces the model to group at least 2-5 samples together.
                    'regressor__min_samples_leaf': {
                        'best': 6,
                        'space': Integer(2, 15),
                        'desc': 'Min samples per leaf'
                    },
                    
                    # 4. NOISE FILTERING: Crucial for your 84 features.
                    # 'sqrt' = looks at only ~9 features per split.
                    # '0.3' = looks at ~25 features per split.
                    'regressor__max_features': {
                        'best': 'sqrt',
                        'space': ['sqrt', 'log2', 0.1, 0.3, 0.5, 0.7, 1.0], 
                        'desc': 'Max features per split'
                    }
                },
                'user': {}, 
                'path': {} 
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