
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

## Independent Tuned SVRs

--- Tuning Target 1/14: Balance Test ---
Best Score (MSE): 1.5423
Best Params: {'pca__n_components': 5, 'regressor__C': 0.044433664965605184, 'regressor__epsilon': 0.6171359284490464, 'regressor__gamma': 'scale', 'regressor__kernel': 'rbf'}

--- Tuning Target 2/14: Single Leg Stance ---
Best Score (MSE): 1.1288
Best Params: {'pca__n_components': 25, 'regressor__C': 0.576414019242899, 'regressor__epsilon': 0.03337932496459199, 'regressor__gamma': 'auto', 'regressor__kernel': 'rbf'}

--- Tuning Target 3/14: Robotrainer Front ---
Best Score (MSE): 0.8509
Best Params: {'pca__n_components': 19, 'regressor__C': 100.0, 'regressor__epsilon': 1.0, 'regressor__gamma': 'auto', 'regressor__kernel': 'rbf'}

--- Tuning Target 4/14: Robotrainer Left ---
Best Score (MSE): 1.0381
Best Params: {'pca__n_components': 34, 'regressor__C': 0.017044779006616908, 'regressor__epsilon': 0.04836897656980925, 'regressor__gamma': 'scale', 'regressor__kernel': 'rbf'}

--- Tuning Target 5/14: Robotrainer Right ---
Best Score (MSE): 0.7449
Best Params: {'pca__n_components': 54, 'regressor__C': 100.0, 'regressor__epsilon': 0.01, 'regressor__gamma': 'scale', 'regressor__kernel': 'rbf'}

--- Tuning Target 6/14: Hand Grip Left ---
Best Score (MSE): 0.7898
Best Params: {'pca__n_components': 54, 'regressor__C': 0.47108828663039926, 'regressor__epsilon': 0.01, 'regressor__gamma': 'auto', 'regressor__kernel': 'rbf'}

--- Tuning Target 7/14: Hand Grip Right ---
Best Score (MSE): 0.3988
Best Params: {'pca__n_components': 56, 'regressor__C': 18.401823963533182, 'regressor__epsilon': 0.01, 'regressor__gamma': 'scale', 'regressor__kernel': 'rbf'}

--- Tuning Target 8/14: Jump & Reach ---
Best Score (MSE): 0.9994
Best Params: {'pca__n_components': 40, 'regressor__C': 15.681962615294953, 'regressor__epsilon': 0.7719310946196394, 'regressor__gamma': 'auto', 'regressor__kernel': 'rbf'}

--- Tuning Target 9/14: Tandem Walk ---
Best Score (MSE): 1.2515
Best Params: {'pca__n_components': 21, 'regressor__C': 0.01778711589862346, 'regressor__epsilon': 0.01, 'regressor__gamma': 'auto', 'regressor__kernel': 'rbf'}

--- Tuning Target 10/14: Figure 8 Walk ---
Best Score (MSE): 1.1808
Best Params: {'pca__n_components': 28, 'regressor__C': 0.016268101934330387, 'regressor__epsilon': 0.01, 'regressor__gamma': 'scale', 'regressor__kernel': 'rbf'}

--- Tuning Target 11/14: Jumping Sideways ---
Best Score (MSE): 1.0240
Best Params: {'pca__n_components': 34, 'regressor__C': 0.017044779006616908, 'regressor__epsilon': 0.04836897656980925, 'regressor__gamma': 'scale', 'regressor__kernel': 'rbf'}

--- Tuning Target 12/14: Throwing Beanbag at Target ---
Best Score (MSE): 1.1883
Best Params: {'pca__n_components': 18, 'regressor__C': 0.01, 'regressor__epsilon': 0.015731213257540094, 'regressor__gamma': 'scale', 'regressor__kernel': 'rbf'}

--- Tuning Target 13/14: Tapping Test ---
Best Score (MSE): 1.1612
Best Params: {'pca__n_components': 37, 'regressor__C': 0.07817726661294962, 'regressor__epsilon': 0.01, 'regressor__gamma': 'auto', 'regressor__kernel': 'rbf'}

--- Tuning Target 14/14: Ruler Drop Test ---
Best Score (MSE): 1.0537
Best Params: {'pca__n_components': 23, 'regressor__C': 0.3753988814338626, 'regressor__epsilon': 0.01, 'regressor__gamma': 'auto', 'regressor__kernel': 'rbf'}

--- Validation Results (CV) ---
RMSE (Scaled Units): 0.9791
RMSE (Real Units):   21.3086

--- Clinical Scale Prediction Ranking (Validation RMSE) ---
Figure 8 Walk                      : 0.6140
Tandem Walk                        : 2.2783
Tapping Test                       : 2.8674
Single Leg Stance                  : 4.3698
Ruler Drop Test                    : 4.3884
Throwing Beanbag at Target         : 4.9440
Hand Grip Right                    : 5.7117
Jumping Sideways                   : 7.0261
Hand Grip Left                     : 7.3916
Jump & Reach                       : 7.4704
Balance Test                       : 14.1833
Robotrainer Right                  : 37.7893
Robotrainer Left                   : 41.9065
Robotrainer Front                  : 51.9846

--- Test Set Results ---
RMSE (Scaled Units): 0.8913
RMSE (Real Units):   24.2923

