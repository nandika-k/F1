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

# Load similar races data
sim_dfs = []
for fp in similar_fps:
    if os.path.exists(fp):
        sim_dfs.append(pd.read_csv(fp))
    else:
        print(f"[warn] similar track file missing, skipping: {fp}")

# Combine similar tracks from sim_dfs
sim_hist = pd.concat(sim_dfs, ignore_index=True)

# Combine dfs for all historical and similar races
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

# Add predictions to abu_avg dataframe
abu_avg['Predicted Finish'] = pred_finish_abu
# Assign each driver a rank and sort by predicted finish
abu_avg['Predicted Rank'] = abu_avg['Predicted Finish'].rank(method='dense', ascending=True).astype(int)
abu_ranking = abu_avg.sort_values('Predicted Rank').reset_index(drop=True)

# Output results
print("\nPredicted Abu Dhabi 2025 Results")
print(abu_ranking[['Driver', 'Predicted Finish', 'Predicted Rank']])

# Save output
abu_ranking[['Driver', 'Predicted Finish', 'Predicted Rank']].to_csv(predict_2025_fp, index=False)