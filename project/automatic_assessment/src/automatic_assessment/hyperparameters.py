import pandas as pd
from skopt.space import Real, Categorical, Integer

class HyperparameterManager:
    """
    Manages default hyperparameters and search spaces for Bayesian Optimization.
    Uses a nested dictionary structure: Model -> Defaults/Dataset -> Params.
    """
    
    def __init__(self, model_type: str, dataset_type: str, n_features: int):
        """
        Args:
            model_type: 'svr' or 'rf'.
            dataset_type: 'user', 'path', or time interval string.
            n_features: Number of features in the dataset (used to cap PCA components).
        """
        self.model_type = model_type
        self.dataset_type = dataset_type
        self.n_features = n_features
            
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
                'space': Integer(5, self.n_features),
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
                'best': 15
                }
            },
            'path': {
                # Specific overrides for Path-based dataset (Large N)
                'pca__n_components': {'best': 38},
                'regressor__estimator__C': {'best': 0.28}
            },
            '1s': {
                'pca__n_components': {'best': 48},
                'regressor__estimator__C': {'best': 0.029623126884685373},
                'regressor__estimator__epsilon': {'best': 0.08330593919633633},
                'regressor__estimator__gamma': {'best': 'scale'},
                'regressor__estimator__kernel': {'best': 'rbf'}
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
            'path': {},
            '1s': {
                'regressor__max_depth': {'best': 10},
                'regressor__max_features': {'best': 'log2'},
                'regressor__min_samples_leaf': {'best': 9},
                'regressor__n_estimators': {'best': 50}
            }
            },
            'mlp': {
            'defaults': {
                # 1. PREPROCESSING: Constrain PCA strictly
                # With 560 samples, keeping > 30 dims usually introduces more noise than signal.
                'pca__n_components': {
                'best': 15,
                'space': Integer(5, self.n_features), 
                'desc': 'Number of PCA components'
                },

                # 2. ARCHITECTURE: The "Bottleneck" Strategy
                # We need narrow layers. 
                # (30,) = ~600 params (Safe). 
                # (100, 50) = ~7000 params (Dangerous).
                'regressor__hidden_layer_sizes': {
                'best': "(32,)", 
                'space': Categorical([
                    "(16,)",          # Very conservative
                    "(32,)",          # Standard for small data
                    "(64,)",          # Moderate
                    "(32, 16)",       # Deep but narrow (bottleneck)
                    "(32, 32)",       # Wider but still safe
                    "(64, 32)",        # Maximum complexity allowed
                    "(16, 16, 16)",     # Deepest safe option
                    "(32, 32, 16)"      # Deep bottleneck
                ]),
                'desc': 'Hidden layer configuration'
                },

                # 3. REGULARIZATION: This is your safety net.
                # Since N is small, we need HIGH Alpha (L2 penalty) to prevent weights from exploding.
                # Shifted range from [1e-5, 1e-1] to [1e-3, 10.0]
                'regressor__alpha': {
                'best': 0.1,
                'space': Real(0.0001, 10.0, prior='log-uniform'),
                'desc': 'L2 regularization strength'
                },

                # 4. SOLVER: Stick to LBFGS
                # Adam is for massive datasets (stochastic). 
                # LBFGS approximates the Hessian matrix and is mathematically superior for < 10k samples.
                'regressor__solver': {
                'best': 'lbfgs',
                'space': Categorical(['lbfgs']), # 'adam', 'sgd'
                'desc': 'Solver (LBFGS is standard for small N)'
                },

                # 5. ACTIVATION
                # Tanh is often better for sensor/physics data as it is zero-centered (-1 to 1).
                # ReLU (0 to inf) is standard but can result in "dead neurons" on small datasets.
                'regressor__activation': {
                'best': 'tanh',
                'space': Categorical(['tanh', 'relu']),
                'desc': 'Activation function'
                },
                
                # 6. ITERATIONS
                # LBFGS needs room to converge.
                'regressor__max_iter': {
                'best': 2000,
                'space': Integer(1000, 5000),
                'desc': 'Max iterations for convergence'
                }
                    },
            'user': {},
            'path': {},
            '1s': {
                # 1. SOLVER: CRITICAL CHANGE
                # Switch to 'adam' for 8k samples. It scales better and allows plotting.
                'regressor__solver': {
                    'best': 'adam',
                    'space': Categorical(['adam']), 
                    'desc': 'Optimizer (Adam is best for N > 3000)'
                },

                # 2. ARCHITECTURE: Slightly Deeper
                # With 8k samples, a single layer of 32 is likely underfitting (too simple).
                # We can now afford 2 layers to capture non-linear interactions.
                'regressor__hidden_layer_sizes': {
                    'best': "(128, 64)",
                    'space': Categorical([
                        # "(8,)",
                        # "(16,)",          # Very conservative
                        # "(32,)",          # Standard for small data
                        # "(50,)",           # Simple baseline
                        # "(64,)",          # Moderate
                        # "(100,)",          # Wide baseline
                        "(32, 16)",       # Deep but narrow (bottleneck)
                        "(32, 32)",       # Wider but still safe
                        "(64, 32)",        # The standard funnel
                        "(128, 64)",       # Complex (Risk of overfitting, but possible)
                        "(16, 16, 16)",     # Deepest safe option
                        "(64, 32, 16)",     # Deep bottleneck
                        "(128, 64, 32)"     # Deep bottleneck
                    ]),
                    'desc': 'Hidden layer configuration'
                },

                # 3. ALPHA: Relaxing the Safety Net
                # With more data, we can lower alpha (allow the model to learn more).
                # Range moves from [0.001 - 10.0] down to [0.0001 - 1.0]
                'regressor__alpha': {
                    'best': 0.001,
                    'space': Real(0.000001, 0.001, prior='log-uniform'),
                    'desc': 'L2 Regularization'
                },

                # 4. LEARNING RATE (Crucial for Adam)
                # If the model "struggles to converge," the learning rate is often too high or low.
                'regressor__learning_rate_init': {
                    'best': 0.0000016,
                    'space': Real(1e-6, 1e-3, prior='log-uniform'),
                    'desc': 'Step size for Adam'
                },

                # 5. BATCH SIZE (New)
                # How many samples the model sees at once. 
                # Smaller batch = more noise/exploration. Larger batch = stable gradient.
                'regressor__batch_size': {
                    'best': 8,
                    'space': Categorical([8, 16, 32, 64, 128]),
                    'desc': 'Mini-batch size'
                },
                
                # 6. PCA
                # With 8000 rows and 62 features, you might NOT need PCA anymore.
                # Your data density is better. Try allowing "None" (using raw features).
                'pca__n_components': {
                    'best': 16, # or None
                    'space': Integer(10, 50), # Search roughly half your feature count
                    'desc': 'PCA Components'
                },

                # 6. ITERATIONS
                # LBFGS needs room to converge.
                'regressor__max_iter': {
                'best': 1043,
                'space': Integer(700, 3000),
                'desc': 'Max iterations for convergence'
                }
            }
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