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
