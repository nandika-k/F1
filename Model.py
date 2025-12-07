import os
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


# Filepaths for race data
race23_fp = "race_data/Abu_Dhabi_2023_Driver_Data.csv"
race24_fp = "race_data/Abu_Dhabi_2024_Driver_Data.csv"

similar_fps = [
    "race_data/Bahrain_2025_Driver_Data.csv",
    "race_data/Canada_2025_Driver_Data.csv",
    "race_data/Saudi_Arabia_2025_Driver_Data.csv",
    "race_data/Singapore_2025_Driver_Data.csv",
    "race_data/Spain_2025_Driver_Data.csv",
]

qualifiying_25 = "race_data/Abu_Dhabi_2025_Qualifying_Data.csv"

predict_2025_fp = "race_data/PREDICTION_Abu_Dhabi_2025_Driver_Data.csv"  # optional

numeric_base_features = [
    'Fastest Lap Time', 'Average Lap Time', 'STD Lap Time',
    'Sector 1 Average', 'Sector 2 Average', 'Sector 3 Average',
    'Qualifying Position'
]
finish_col = 'Finish Position'

# Weights for 2023 & 2024 races
weight_23 = 0.5
weight_24 = 0.7

# Weights for Abu Dhabi races vs similar tracks
abu_weight = 1.0
sim_weight = 0.2

# Load Abu Dhabi race data
race23 = pd.read_csv(race23_fp)
race24 = pd.read_csv(race24_fp)
quali25 = pd.read_csv(qualifiying_25)

# Load similar races data
sim_dfs = []
for fp in similar_fps:
    if os.path.exists(fp):
        sim_dfs.append(pd.read_csv(fp))
    else:
        print(f"[warn] similar track file missing, skipping: {fp}")

# Combine similar tracks from sim_dfs
sim_hist = pd.concat(sim_dfs, ignore_index=True)


# Filter to only include current drivers (apply per-DataFrame)
current_drivers = ['PIA', 'VER', 'HAM', 'LEC', 'SAI', 'RUS', 'NOR', 'ALO', 'GAS', 'HUL', 'TSU', 'STR', 'ALB']
race23 = race23[race23['Driver'].isin(current_drivers)].reset_index(drop=True)
race24 = race24[race24['Driver'].isin(current_drivers)].reset_index(drop=True)
sim_hist = sim_hist[sim_hist['Driver'].isin(current_drivers)].reset_index(drop=True)

# Alls the dfs for historical and similar races
all_dfs_for_laps = [race23, race24] + sim_dfs

# Create feature list, same as numeric for right now
features = numeric_base_features

# Merge Abu Dhabi 23 + 24 on Driver
abu_hist = pd.merge(
    race23[["Driver"] + features + [finish_col]],
    race24[["Driver"] + features + [finish_col]],
    on="Driver",
    suffixes=('_23', '_24')
)
# Create abu_avg df using weighted average of each feature across 23 & 24
abu_avg = pd.DataFrame()
abu_avg['Driver'] = abu_hist['Driver']

for f in features:
    col23 = f + "_23"
    col24 = f + "_24"
    abu_avg[f] = weight_23 * abu_hist[col23].values + weight_24 * abu_hist[col24].values

# Weighted finish label
abw_col_23 = finish_col + '_23'
abw_col_24 = finish_col + '_24'
if abw_col_23 in abu_hist.columns and abw_col_24 in abu_hist.columns:
    abu_hist['Weighted Finish'] = weight_23 * abu_hist[abw_col_23] + weight_24 * abu_hist[abw_col_24]
else:
    # fallback if something unexpected happened
    abu_hist['Weighted Finish'] = abu_hist.get(finish_col + '_24', abu_hist.get(finish_col + '_23', np.nan))

abu_avg['Weighted Finish'] = abu_hist['Weighted Finish']

# Similar tracks' features define
sim_features = sim_hist[features].copy()
sim_features['Weighted Finish'] = sim_hist[finish_col].values

# Combine Abu Dhabi + similar track features for scaling and training
X_features = pd.concat([abu_avg[features], sim_features[features]], ignore_index=True)
# Combine labels or Y value for both sets of features
labels = pd.concat([abu_avg['Weighted Finish'], sim_features['Weighted Finish']], ignore_index=True)

# Handle NaN values
max_pos = int(max(labels.dropna().max() if not labels.dropna().empty else 20, 20))
X_features = X_features.fillna(max_pos + 1)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features.values)

# Split back into abu_scaled and sim_scaled (useful for individual inspection)
abu_scaled = X_scaled[:len(abu_avg)]
sim_scaled = X_scaled[len(abu_avg):]

# Weight according to weights decided above
weights = np.concatenate([
    np.full(len(abu_scaled), abu_weight),
    np.full(len(sim_scaled), sim_weight)
])

# Train Neural Network
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=5000,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=42
)
model.fit(X_scaled, labels.values, sample_weight=weights)

pred_finish_abu = model.predict(abu_scaled)

# # Predict results
# # Merge current qualifying grid into abu_avg 
# abu_avg = abu_avg.merge(quali25[['Driver', 'Qualifying_Position_25']], on='Driver', how='left')
# new_features = features + ['Qualifying_Position_25']

# # rebuild abu features and scale using the previously-fitted scaler
# abu_features_current = abu_avg[features].fillna(max_pos + 1)
# abu_scaled_current = scaler.transform(abu_features_current.values)
# pred_finish_abu = model.predict(abu_scaled_current)

# Map qualifying positions into abu_avg by Driver and compute a safe z-score
qual_series = quali25.set_index('Driver')['Qualifying Position'] if 'Qualifying Position' in quali25.columns else None
if qual_series is None:
    # try alternative column names
    for c in ['Position', 'Grid', 'QualPos', 'Qualifying_Position_25']:
        if c in quali25.columns:
            qual_series = quali25.set_index('Driver')[c]
            break

if qual_series is None:
    # no qualifying data found; fall back to zeros
    qual_mapped = pd.Series(np.full(len(abu_avg), max_pos + 1), index=abu_avg.index)
else:
    qual_mapped = abu_avg['Driver'].map(qual_series).astype(float).fillna(max_pos + 1)

# z-score (guard against zero std)
qual_std = qual_mapped.std()
if pd.isna(qual_std) or qual_std == 0:
    qual_scaled = (qual_mapped - qual_mapped.mean()).fillna(0.0)
else:
    qual_scaled = (qual_mapped - qual_mapped.mean()) / qual_std

# Align predictions and qual by abu_avg order and compute blended prediction
alpha = 0.3
pred_array = np.asarray(pred_finish_abu)
qual_array = np.asarray(qual_scaled)
if pred_array.shape[0] != qual_array.shape[0]:
    # if lengths mismatch, try aligning by driver index
    pred_series = pd.Series(pred_array, index=abu_avg['Driver'].values)
    qual_array = abu_avg['Driver'].map(qual_series).astype(float).fillna(max_pos + 1).values
abu_avg['Predicted Finish'] = (1 - alpha) * pred_array + alpha * qual_array * max_pos

# Assign each driver a rank and sort by predicted finish
abu_avg['Predicted Rank'] = abu_avg['Predicted Finish'].rank(method='dense', ascending=True).astype(int)
abu_ranking = abu_avg.sort_values('Predicted Rank').reset_index(drop=True)

# Output results
print("\nPredicted Abu Dhabi 2025 Results")
print(abu_ranking[['Driver', 'Predicted Rank']])

# Save output
abu_ranking[['Driver', 'Predicted Rank']].to_csv(predict_2025_fp, index=False)