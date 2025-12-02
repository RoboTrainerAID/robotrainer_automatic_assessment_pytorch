# Architecture descicions

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
