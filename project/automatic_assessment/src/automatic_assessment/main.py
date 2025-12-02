import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from automatic_assessment.datasets import Dataset
from automatic_assessment.models import Model
from automatic_assessment.imputers import SmartImputer

def main():
    # --- CONFIGURATION ---
    DATASET_TYPE = 'path' # Options: 'user', 'path', '1s', '500ms', '100ms'
    PERFORM_TUNING = False
    RECREATE_DATASET = False
    MODEL_TYPE = 'rf' # Options: 'svr', 'rf'
    
    # --- 1. Initialize Dataset ---
    dataset = Dataset(dataset_type=DATASET_TYPE, recreate=RECREATE_DATASET)
    print(f"Dataset aggregated per {DATASET_TYPE}")

    # --- 2. Prepare Data ---
    X_train_val, X_test, y_train_val, y_test, users_train_val, users_test = dataset.get_train_test_split()
    
    print(f"Train/Val Samples: {len(X_train_val)} (Users: {len(np.unique(users_train_val))})")
    print(f"Test Samples: {len(X_test)} (Users: {len(np.unique(users_test))})")
    print(f"Number of Features: {X_train_val.shape[1]}")

    # --- 3. Initialize Model ---
    model = Model.create(
        model_type=MODEL_TYPE,
        dataset=dataset
    )

    # --- 4. Build Pipeline ---
    # Define the pipeline structure here for full control
    if MODEL_TYPE == 'svr':
        regressor = MultiOutputRegressor(SVR())
        steps = [
            ('imputer', SmartImputer(strategy='user_mean')),
            # ('imputer', SmartImputer(strategy='global_mean')),
            # ('imputer', SmartImputer(strategy='path_mean')),
            # ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('pca', PCA()),
            ('regressor', regressor)
        ]
    elif MODEL_TYPE == 'rf':
        # RF: Native Multi-output, No Scaling, No PCA (for feature importance)
        regressor = RandomForestRegressor(random_state=0)
        steps = [
            ('imputer', SmartImputer(strategy='global_mean')),
            ('regressor', regressor)
        ]
    else:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

    base_pipeline = Pipeline(steps)

    # --- 5. Hyperparameter Tuning or Loading ---
    # Pre-scale Y for tuning (SVR is sensitive to target scale)
    y_scaler_tune = StandardScaler()
    y_train_val_scaled = y_scaler_tune.fit_transform(y_train_val)

    if PERFORM_TUNING:
        # Pass the pipeline template and data; Model handles CV/fit_params internally via Dataset
        model.tune(X_train_val, y_train_val_scaled, users_train_val, base_pipeline, n_iter=30)
    else:
        print(f"\n--- Skipping Hyperparameter Tuning (Using Hardcoded {MODEL_TYPE.upper()} Parameters) ---")
        # Apply default parameters to the pipeline
        model.set_pipeline(base_pipeline)

    # --- 6. Evaluation (LOOCV) ---
    splitter = dataset.get_cv_splitter(X_train_val, y_train_val, users_train_val)
    
    scaled_mse_scores = [] 
    real_mse_scores = []
    plot_data = []
    target_cols = dataset.TARGET_COLS
    
    # Scaler for Y (used for evaluation)
    y_scaler = StandardScaler()

    print("Starting Detailed LOOCV Evaluation...")

    for train_index, val_index in splitter:
        # A. Split
        X_t, X_v = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
        y_t, y_v = y_train_val[train_index], y_train_val[val_index]
        
        current_user_id = users_train_val[val_index][0]

        # B. Scale Targets (Y)
        y_t_scaled = y_scaler.fit_transform(y_t)
        y_v_scaled = y_scaler.transform(y_v) 
        
        # C. Train Model (Pipeline handles X scaling/imputing/PCA)
        model.train(X_t, y_t_scaled)
        
        # D. Predict
        y_pred_scaled_raw = model.predict(X_v)
        
        # AGGREGATION LOGIC
        if dataset.has_multiple_rows_per_user:
            y_pred_scaled = np.mean(y_pred_scaled_raw, axis=0, keepdims=True)
            y_v_single = y_v[0:1] 
            y_v_scaled_single = y_v_scaled[0:1]
        else:
            y_pred_scaled = y_pred_scaled_raw
            y_v_single = y_v
            y_v_scaled_single = y_v_scaled
        
        # E. Calculate Error
        mse_scaled = mean_squared_error(y_v_scaled_single, y_pred_scaled)
        scaled_mse_scores.append(mse_scaled)
        
        y_pred_real = y_scaler.inverse_transform(y_pred_scaled)
        mse_real = mean_squared_error(y_v_single, y_pred_real)
        real_mse_scores.append(mse_real)

        # F. Collect Data
        for i, col in enumerate(target_cols):
            plot_data.append({
                'User': current_user_id,
                'Task': col,
                'Actual Real': y_v_single[0][i],
                'Predicted Real': y_pred_real[0][i],
                'Absolute Error Real': abs(y_v_single[0][i] - y_pred_real[0][i]),
                'Set': 'Validation'
            })

    print("\n--- Validation Results (CV) ---")
    print(f"RMSE (Scaled Units): {np.sqrt(np.mean(scaled_mse_scores)):.4f}")
    print(f"RMSE (Real Units):   {np.sqrt(np.mean(real_mse_scores)):.4f}")

    # --- 7. Final Test Set Evaluation ---
    # Train on full training set
    y_final_scaled = y_scaler.fit_transform(y_train_val)
    model.train(X_train_val, y_final_scaled)
    
    # Predict on test set
    y_test_scaled = y_scaler.transform(y_test)
    y_pred_test_raw = model.predict(X_test)
    
    # Aggregation for Test Set
    unique_test_users = np.unique(users_test)
    test_mse_real_list = []
    test_mse_scaled_list = []
    
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

    # Feature Importance (RF Only)
    if MODEL_TYPE == 'rf':
        print("\n--- Feature Importance ---")
        # Access the regressor step directly
        rf_model = model.pipeline.named_steps['regressor']
        
        if hasattr(rf_model, 'feature_importances_'):
            importances = rf_model.feature_importances_
            
            feature_names = []
            # Retrieve feature names from the imputer to account for dropped columns
            if 'imputer' in model.pipeline.named_steps:
                imputer = model.pipeline.named_steps['imputer']
                # Robust way: use feature_names_in_ and filter out cols_to_drop_
                if hasattr(imputer, 'feature_names_in_') and hasattr(imputer, 'cols_to_drop_'):
                    feature_names = [f for f in imputer.feature_names_in_ if f not in imputer.cols_to_drop_]
                elif hasattr(imputer, 'get_feature_names_out'):
                    feature_names = imputer.get_feature_names_out()
            
            if not feature_names:
                feature_names = dataset.feature_names
            
            # Ensure lengths match before creating DataFrame
            if len(feature_names) == len(importances):
                fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                fi_df = fi_df.sort_values(by='Importance', ascending=False)
                
                print(fi_df.head(15))
                fi_df.to_csv("/data/feature_importance.csv", index=False)
                print("Feature importance saved to /data/feature_importance.csv")
            else:
                print(f"Warning: Feature names count ({len(feature_names)}) does not match importance count ({len(importances)}). Skipping table.")

    # Save Results
    results_df = pd.DataFrame(plot_data)
    results_path = "/data/loocv_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nDetailed results saved to {results_path}")

if __name__ == "__main__":
    main()