import fastf1 as FastF1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def qualifying_data(race, year):
    session = FastF1.get_session(year, race, 'Q')
    session.load()

    qualifying_data = {}

    for row in session.results.itertuples():
        driver = row.Abbreviation
        position = row.Position
        qualifying_data[driver] = position

    # print(race, year, '\n', qualifying_data)
    return qualifying_data


def generate_session_csv(race, year, session, csv_name):
    laps_df = pd.DataFrame()
    quali_data = qualifying_data(race, year)

    for driver in session.drivers:
        driver_laps = session.laps.pick_drivers(driver)
        driver_laps_df = pd.DataFrame(driver_laps)

        #drop if position is NaT
        driver_laps_df = driver_laps_df.dropna(subset=['Position'])

        if not driver_laps_df.empty:
            row = {
                'Driver': driver_laps_df['Driver'].iloc[0],
                'Fastest Lap Time': min(driver_laps_df['LapTime'].dt.total_seconds()),
                'Average Lap Time': np.mean(driver_laps_df['LapTime'].dt.total_seconds()),
                'STD Lap Time': np.std(driver_laps_df['LapTime'].dt.total_seconds()),
                'Sector 1 Average': np.mean(driver_laps_df['Sector1Time'].dt.total_seconds()),
                'Sector 2 Average': np.mean(driver_laps_df['Sector2Time'].dt.total_seconds()),
                'Sector 3 Average': np.mean(driver_laps_df['Sector3Time'].dt.total_seconds()),
                'Qualifying Position': quali_data.get(driver_laps_df['Driver'].iloc[0], None),
                'Finish Position': driver_laps_df['Position'].iloc[-1]
            }
            laps_df = laps_df._append(row, ignore_index=True)

    # Ensure output directory exists and write file into `race_data/` folder
    out_dir = 'race_data'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(csv_name))
    laps_df.to_csv(out_path, index=False)

def main():
    # Enable FastF1 cache
    FastF1.Cache.enable_cache('cache')

    session1 = FastF1.get_session(2023, 'Abu Dhabi', 'R')
    session1.load()

    session2 = FastF1.get_session(2024, 'Abu Dhabi', 'R')
    session2.load()

    generate_session_csv('Abu Dhabi', 2023, session1, 'Abu_Dhabi_2023_Driver_Data.csv')
    generate_session_csv('Abu Dhabi', 2024, session2, 'Abu_Dhabi_2024_Driver_Data.csv')

    for race in ['Spain', 'Canada', 'Bahrain', 'Saudi Arabia', 'Singapore']:
        session = FastF1.get_session(2025, race, 'R')
        session.load()
        race = race.replace(" ", "_")
        generate_session_csv(race, 2025, session, f'{race}_2025_Driver_Data.csv')

if __name__ == "__main__":
    main()