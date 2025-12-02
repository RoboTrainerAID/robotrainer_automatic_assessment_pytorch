import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV

from automatic_assessment.datasets import Dataset
from automatic_assessment.hyperparameters import HyperparameterManager

def main():
    # --- CONFIGURATION ---
    DATASET_TYPE = 'path' # Options: 'user', 'path', '1s', '100ms', '0.5s'
    PERFORM_TUNING = False
    RECREATE_DATASET = False
    MODEL_TYPE = 'svr' # Options: 'svr', 'rf' (future)
    
    # --- 1. Initialize Dataset ---
    dataset = Dataset(dataset_type=DATASET_TYPE, recreate=RECREATE_DATASET)
    print(f"Dataset aggregated per {DATASET_TYPE}")

    # --- 2. Prepare Data ---
    X_train_val, X_test, y_train_val, y_test, users_train_val, users_test = dataset.get_train_test_split()
    
    print(f"Train/Val Samples: {len(X_train_val)} (Users: {len(np.unique(users_train_val))})")
    print(f"Test Samples: {len(X_test)} (Users: {len(np.unique(users_test))})")
    print(f"Number of Features: {X_train_val.shape[1]}")

    # --- 3. Initialize Hyperparameter Manager ---
    hp_manager = HyperparameterManager(
        model_type=MODEL_TYPE,
        dataset_type=dataset.type,
        n_features=X_train_val.shape[1],
        n_samples=X_train_val.shape[0]
    )

    # --- 4. Hyperparameter Tuning or Loading ---
    
    if PERFORM_TUNING:
        print(f"\n--- Starting Hyperparameter Tuning ({MODEL_TYPE.upper()}) ---")
        
        # Define Pipeline for Tuning
        # We include Imputer, Scaler, PCA, and Regressor to tune the whole flow
        tuning_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('pca', PCA()),
            ('regressor', MultiOutputRegressor(SVR())) # Default estimator, swapped if needed
        ])

        # Get dynamic search space from manager
        search_spaces = hp_manager.get_search_space()

        # Pre-scale Y for tuning (SVR is sensitive to target scale)
        # The pipeline handles X scaling, but we must scale Y manually for the tuner
        y_scaler_tune = StandardScaler()
        y_train_val_scaled = y_scaler_tune.fit_transform(y_train_val)

        # Get CV strategy and fit parameters from the dataset class
        cv_tuning = dataset.get_cv_strategy()
        fit_params = dataset.get_fit_params(X_train_val, y_train_val_scaled, users_train_val)

        opt = BayesSearchCV(
            estimator=tuning_pipeline,
            search_spaces=search_spaces,
            n_iter=50, # Number of parameter settings that are sampled
            cv=cv_tuning,
            n_jobs=-1, 
            verbose=1,
            random_state=0,
            scoring='neg_mean_squared_error'
        )

        # Fit using the polymorphic fit_params (handles groups automatically)
        opt.fit(X_train_val, y_train_val_scaled, **fit_params)

        print(f"Best CV Score (MSE): {-opt.best_score_:.4f}")
        print("Best Parameters found:", opt.best_params_)

        best_params = opt.best_params_
    else:
        print(f"\n--- Skipping Hyperparameter Tuning (Using Hardcoded {MODEL_TYPE.upper()} Parameters) ---")
        best_params = hp_manager.get_best_params()

    # Extract params for final model
    best_n_components = best_params['pca__n_components']
    best_C = best_params['regressor__estimator__C']
    best_epsilon = best_params['regressor__estimator__epsilon']
    best_gamma = best_params['regressor__estimator__gamma']
    best_kernel = best_params['regressor__estimator__kernel']
    
    # --- 5. Final Model Initialization ---
    print(f"\nInitializing Model with: C={best_C}, epsilon={best_epsilon}, PCA={best_n_components}")
    
    imputer = SimpleImputer(strategy='mean')
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    pca = PCA(n_components=best_n_components)
    svr_base = SVR(kernel=best_kernel, C=best_C, epsilon=best_epsilon, gamma=best_gamma)
    multi_svr = MultiOutputRegressor(svr_base)

    # --- 6. Evaluation (LOOCV) ---
    # Get CV splitter directly from dataset (handles groups internally)
    splitter = dataset.get_cv_splitter(X_train_val, y_train_val, users_train_val)
    
    scaled_mse_scores = [] 
    real_mse_scores = []
    plot_data = []
    target_cols = dataset.TARGET_COLS

    print("Starting Detailed LOOCV Evaluation...")

    for train_index, val_index in splitter:
        # A. Split
        X_t, X_v = X_train_val[train_index], X_train_val[val_index]
        y_t, y_v = y_train_val[train_index], y_train_val[val_index]
        
        # Get Real User ID for this validation sample
        # In LOGO, val_index covers all rows for that user
        current_user_id = users_train_val[val_index][0]

        # B. Impute & Scale Inputs (X)
        # Fit imputer on training split only to prevent leakage
        X_t_imputed = imputer.fit_transform(X_t)
        X_v_imputed = imputer.transform(X_v)

        X_t_scaled = x_scaler.fit_transform(X_t_imputed)
        X_v_scaled = x_scaler.transform(X_v_imputed)
        
        # C. Scale Targets (Y)
        y_t_scaled = y_scaler.fit_transform(y_t)
        y_v_scaled = y_scaler.transform(y_v) 
        
        # D. PCA
        X_t_pca = pca.fit_transform(X_t_scaled)
        X_v_pca = pca.transform(X_v_scaled)
        
        # E. Train
        multi_svr.fit(X_t_pca, y_t_scaled)
        
        # F. Predict
        y_pred_scaled_raw = multi_svr.predict(X_v_pca)
        
        # AGGREGATION LOGIC
        if dataset.has_multiple_rows_per_user:
            # We have multiple predictions for the same user (one per path)
            # We average them to get the final user assessment
            y_pred_scaled = np.mean(y_pred_scaled_raw, axis=0, keepdims=True)

            # We also need to compare against a single actual value per user
            # Since y_v contains identical rows for the same user, we take the first one
            y_v_single = y_v[0:1] 
            y_v_scaled_single = y_v_scaled[0:1]
        else:
            y_pred_scaled = y_pred_scaled_raw
            y_v_single = y_v
            y_v_scaled_single = y_v_scaled
        
        # G. Calculate Scaled Error
        mse_scaled = mean_squared_error(y_v_scaled_single, y_pred_scaled)
        scaled_mse_scores.append(mse_scaled)
        
        # H. Inverse Transform & Calculate Real Error
        y_pred_real = y_scaler.inverse_transform(y_pred_scaled)
        mse_real = mean_squared_error(y_v_single, y_pred_real)
        real_mse_scores.append(mse_real)

        # H. Collect Data for Visualization
        for i, col in enumerate(target_cols):
            # Real Values
            actual_real = y_v_single[0][i]
            pred_real_val = y_pred_real[0][i]
            
            # Scaled Values
            actual_scaled = y_v_scaled_single[0][i]
            pred_scaled_val = y_pred_scaled[0][i]
            
            plot_data.append({
                'User': current_user_id,
                'Task': col,
                'Actual Real': actual_real,
                'Predicted Real': pred_real_val,
                'Absolute Error Real': abs(actual_real - pred_real_val),
                'Actual Scaled': actual_scaled,
                'Predicted Scaled': pred_scaled_val,
                'Absolute Error Scaled': abs(actual_scaled - pred_scaled_val),
                'Set': 'Validation'
            })

    print("\n--- Validation Results (CV) ---")
    print(f"RMSE (Scaled Units): {np.sqrt(np.mean(scaled_mse_scores)):.4f}")
    print(f"RMSE (Real Units):   {np.sqrt(np.mean(real_mse_scores)):.4f}")

    # --- 7. Final Test (Real World Values) ---

    # Prepare Full Training Data
    # Impute and Scale on full training set
    X_final_imputed = imputer.fit_transform(X_train_val)
    X_final_scaled = x_scaler.fit_transform(X_final_imputed)
    
    y_final_scaled = y_scaler.fit_transform(y_train_val)
    X_final_pca = pca.fit_transform(X_final_scaled)

    # Prepare Test Data
    # Use the fitted imputer and scaler from full training set
    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = x_scaler.transform(X_test_imputed)
    
    X_test_pca = pca.transform(X_test_scaled)
    y_test_scaled = y_scaler.transform(y_test)

    # Train (Train again on FULL training data)
    multi_svr.fit(X_final_pca, y_final_scaled)

    # Predict (on holdout test set)
    y_pred_test_raw = multi_svr.predict(X_test_pca)
    
    # Aggregation for Test Set
    # We need to aggregate predictions per user manually since X_test is flattened
    unique_test_users = np.unique(users_test)
    
    test_mse_scaled_list = []
    test_mse_real_list = []
    
    for u_id in unique_test_users:
        # Find indices for this user
        u_indices = np.where(users_test == u_id)[0]
        
        # Get predictions for this user's paths
        u_preds_scaled = y_pred_test_raw[u_indices]
        
        # Average them
        u_pred_scaled_mean = np.mean(u_preds_scaled, axis=0, keepdims=True)
        u_pred_real_mean = y_scaler.inverse_transform(u_pred_scaled_mean)
        
        # Get actuals (take first one)
        u_actual_scaled = y_test_scaled[u_indices][0:1]
        u_actual_real = y_test[u_indices][0:1]
        
        # Calculate Error
        test_mse_scaled_list.append(mean_squared_error(u_actual_scaled, u_pred_scaled_mean))
        test_mse_real_list.append(mean_squared_error(u_actual_real, u_pred_real_mean))
        
        # Collect Data
        for i, col in enumerate(target_cols):
            actual_real = u_actual_real[0][i]
            pred_real_val = u_pred_real_mean[0][i]
            abs_err_real = abs(actual_real - pred_real_val)
            
            actual_scaled = u_actual_scaled[0][i]
            pred_scaled_val = u_pred_scaled_mean[0][i]
            abs_err_scaled = abs(actual_scaled - pred_scaled_val)
            
            plot_data.append({
                'User': u_id,
                'Task': col,
                'Actual Real': actual_real,
                'Predicted Real': pred_real_val,
                'Absolute Error Real': abs_err_real,
                'Actual Scaled': actual_scaled,
                'Predicted Scaled': pred_scaled_val,
                'Absolute Error Scaled': abs_err_scaled,
                'Set': 'Test'
            })

    print("\n--- Test Set Results ---")
    print(f"RMSE (Scaled Units): {np.sqrt(np.mean(test_mse_scaled_list)):.4f}")
    print(f"RMSE (Real Units):   {np.sqrt(np.mean(test_mse_real_list)):.4f}")

    # Save Plot Data
    results_df = pd.DataFrame(plot_data)
    results_path = "/data/loocv_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nDetailed results saved to {results_path} for visualization.")

if __name__ == "__main__":
    main()