
- https://medium.com/analytics-vidhya/dimensionality-reduction-techniques-in-machine-learning-9098037baddc

# Architecture descicions

Problems:

- small dataset (N=28 users with 20 paths, ca. 560 total samples)
- high dimensional input data (76 features per timestep, timeseries length varies between ca. 50-300 timesteps)
- noisy input data (human-computer interaction data)
- target concept is hard to predict (clinical scales)
- target data is continuous (regression problem)
- potential overfitting
- high variance between users
- different sampling rates possible (from full timeseries to per-path averages)
- different feature sets possible (interactions only, interactions + demographics, interactions + demographics + task difficulty)
- different output targets possible (single clinical scale, multioutput clinical scales)
- SPARSE data (not all paths have all sensor data, and not all sampling bins of e.g. 1s have all sensor data)
- SimpleImputers try to fill in nans based on global means, which may introduce bias
- Some timeseries like hrv and ppi hav > 10% nans at 1s sampling, depending on the sampling rate even a lot more

Design choices to address problems:

- dimensionality reduction with PCA
- model types that are robust to noise and overfitting (SVR, Random Forest, MLP with regularization)
- hyperparameter tuning with cross-validation and Bayesian optimization

Input data:

- Sampling rate:
  - each timestep with user ID (wahrscheinlich schlechter als timeseries mit LSTM verarbeiten)
  - averaged per person
  - per path
  - per second (Vorteil ich kann unterschiedliche Sekunden sampeln und so die Datenmenge erhöhen)
  - full timeseries
- Features:
  - only interactions data
  - interactions + demographics
  - interactions + demographics + task difficulty
Output data: single clinical scale / all multioutput clinical scales

## Architecture

1. Data Preprocessing
    - Imputing
      - impute with zeros
      - impute with mean/median
      - impute with closest neighbor
    - Augmentation
      - timeseries meta data (e.g. mean, std, skewness, kurtosis, entropy, trend)
      - noise addition
    - Merging

1. Load dataset from exsiting class

    ```python
    dataset = DatasetConv1s(recreate=False)
    X, y, users, feature_names = dataset.get_user_level_dataset()
    ```

2. Scaling
    - X and y StandardScaler
      - How to handle predictions that should be outside of scale 0-1? Can a NN head predict 1.1?
    - Always use scaled values for validation and loss calculation

3. Test Data Split
    - Problem: **Sample bias** (small N, high variance between users, test set performance varies a lot depending on which users are in test set)
    - Solution: Nested CV with outer LOOCV loop over all users and inner LOOCV loop for hyperparameter tuning
    - Goal: Get robust estimate of model performance on unseen users
    - Normal Test set assumption: test set is representative sample of the true population
      - Does not hold here because of sample bias
      - With 4 test users, no guarantee they represent the curve. Statistically "unlucky" if catching the outliers, but then effectively punishing the model for the splitting strategy, not its actual lack of capability.
    - ~~Test set of 4 users held out completely~~
    - ensure no data from same user in train and val
      - automatically handle if multiple samples have the same user ID
      - both cases could occur. E.g. if using per-path data, multiple samples per user exist. If using per-user data, only one sample per user exists.

3. Dimensionality Reduction
    - LASSO
      - How to find the optimal alpha?
      - Can I set a min or max number of features to select?
      - How to handle multioutput and single-target regression?
        - Do it independently for each target inside the CV loop?

4. Train Validation Split
    - Leave-one-out CV (no data from same user in train and val)
      - or Leave-one-group-out CV with user IDs as groups if multiple samples have the same user ID
      - Determine automatically if LOO or LOGO should be used

5. Hyperparameter tuning
    - Bayesian Optimization
    - Use same framework as model train pipeline and loocv loop for tuning
    - Save and load best model parameters in yaml files
    - Optimize per target vs multioutput
    - Easily turn off and on with aingle flag

6. Model
    - separate class for storing all the model information and hyperparameters
    - This should be easily expandable to try out different models
    - Each should have the same interface like a forward(X) method for predictions

7. Model Training
    - separate methods for train_step, train_epoch, validate
    - save val predictions for each fold for later analysis
    - Each fold in loocv equals the score for a single user

8. Model Evaluation
    - Calculate RMSE (everything always scaled values)
    - Save val predictions as pandas dataframe and csv for later analysis
      - Create a new folder for each model (model name saved in respective class, Folder path with global variable)
      - Save hyperparameters used for training in a yaml file in the same folder
      - Save scores (final and per user from each fold) as pandas dataframe and csv for later analysis
      - Save feature importances if possible (e.g. LASSO coefficients)

9. Visualization / Reporting
    - Separate class or set of methods that only operate on the saved csvs and dataframes
    - Compare single-target vs multioutput models with plots and tables
    - Analyse LASSO feature importances if possible
    - Plot loss / learning curves with train and val loss per epoch
    - Performance as parity plot
    - Prepare comparison of different models (although for now only one used) by iterating all the model folders and aggregating
    

# How to start
1. Dataset generation
   1. Place all .bag files in one folder
   2. with docker `robotrainer_docker_meldic/gait` branch gait follow README instructions there to generate gait.bag files
   3. Place all gait.bag files in another folder
   4. with docker `robotrainer_docker_meldic/bag_to_csv` branch `bag_to_csv` 
      1. Check config.ini for correct folder paths and topics to filter out
      2. run `bag_to_numpy_dataset.py` to generate the dataset as .npy files
      3. Copy folder into this docker data folder
   