# F1 Prediction Pipeline

A pipeline that extracts FastF1 session laps, builds race data CSVs grouped by driver, and trains/predicts driver performance for the F1 Abu Dhabi Grand Prix.

**Project structure (high level)**
- `Data_Generator.py` — generates per-race CSV summaries in `race_data/`.
- `Model.py` — loads historical CSVs, preprocesses data, trains a **Neural Network**, and outputs predictions to terminal and a CSV file.
- `race_data/` — generated driver CSVs for races (output directory).
- `cache/` — FastF1 cache for all the raw race data.

## Requirements
See requirements.txt.

## Quick setup
1. Ensure Python 3.13 is on your PATH.
2. Run main.py

## Output
- Predictions are written as a CSV file (e.g., `PREDICTION_Abu_Dhabi_2025_Driver_Data.csv`) inside `race_data/` and to the terminal.