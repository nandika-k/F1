import fastf1 as FastF1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_session_csv(session, csv_name):
    laps_df = pd.DataFrame()
    
    for driver in session.drivers:
        driver_laps = session.laps.pick_drivers(driver)
        driver_laps_df = pd.DataFrame(driver_laps)

        row = {
            'Driver': driver_laps_df['Driver'].iloc[0],
            'Fastest Lap Time': min(driver_laps_df['LapTime'].dt.total_seconds()),
            'Average Lap Time': np.mean(driver_laps_df['LapTime'].dt.total_seconds()),
            'STD Lap Time': np.std(driver_laps_df['LapTime'].dt.total_seconds()),
            'Sector 1 Average': np.mean(driver_laps_df['Sector1Time'].dt.total_seconds()),
            'Sector 2 Average': np.mean(driver_laps_df['Sector2Time'].dt.total_seconds()),
            'Sector 3 Average': np.mean(driver_laps_df['Sector3Time'].dt.total_seconds()),
            'Positions': driver_laps_df['Position'].tolist()
        }
        laps_df = laps_df._append(row, ignore_index=True)

    laps_df.to_csv(csv_name, index=False)

def main():
    # Enable FastF1 cache
    FastF1.Cache.enable_cache('cache')

    session1 = FastF1.get_session(2023, 'Abu Dhabi', 'R')
    session1.load()

    session2 = FastF1.get_session(2024, 'Abu Dhabi', 'R')
    session2.load()

    generate_session_csv(session1, 'Abu_Dhabi_2023_Driver_Data.csv')
    generate_session_csv(session2, 'Abu_Dhabi_2024_Driver_Data.csv')

if __name__ == "__main__":
    main()