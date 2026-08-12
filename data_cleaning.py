import numpy as np
import pandas as pd
import os

def load_raw_data():
    # 1. Load CSV File
    
    abs_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(abs_folder, 'data')

    os.makedirs(data_folder, exist_ok=True)

    route_raw = os.path.join(data_folder, 'raw_data.csv')

    df_raw = pd.read_csv(route_raw)
    
    return df_raw

def initial_inspection(df_raw):
    # 2. Initial Inspection (Nan-Values, Duplicates, Datatypes...)
    
    print('\n---------------- INITIAL INSPECTION OF THE RAW DATA ----------------\n')
    print(df_raw.head(), '\n')
    print(df_raw.info(), '\n')
    print("Missing values per column:\n", df_raw.isnull().sum())


def data_cleaning(df):
    # 3. Data Cleaning Function
    
    df = df.copy()
    
    # A. Delete every duplicate row
    df = df.drop_duplicates()
    
    # B. Delete NaN values from 'Furnace Temperature' & 'Ambient Temperature'
    df = df.dropna(subset = ['Furnace Temperature', 'Ambient Temperature'])
    
    # C. Replace outliers of 'Production Rate' & 'Machine Operation Hours' with their averages
    mask_p = (df['Production Rate'] >= 125) & (df['Production Rate'] <= 175)
    mask_m = (df['Machine Operation Hours'] >= 15) & (df['Machine Operation Hours'] <= 21)
    
    mean_p = df.loc[mask_p, 'Production Rate'].mean()
    mean_m = df.loc[mask_m, 'Machine Operation Hours'].mean()
    
    df.loc[~mask_p, 'Production Rate'] = mean_p
    df.loc[~mask_m, 'Machine Operation Hours'] = mean_m
    
    return df

def save_clean_data(df):
    # 4. Save the Cleaned Data & Print Verification
    
    abs_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(abs_folder, 'data')
    route_clean = os.path.join(data_folder, 'clean_data.csv')

    # Save Cleaned Data
    df.to_csv(route_clean, index=False)
    print(f"\n[SUCCESS] Clean data saved to: {route_clean}")

    # Print Cleaned Data Verification
    print('\n------------------- FINAL INSPECTION CLEANED DATA -------------------\n')
    print(df.head(), '\n')
    print(df.info())
    print("\nMissing values after cleaning:\n", df.isnull().sum())
    
def main():
    print('--- STARTING DATA CLEANING PIPELINE ---')
    
    # Step 1: Load Data
    df_raw = load_raw_data()
    
    # Step 2: Initial Inspection
    initial_inspection(df_raw)
    
    # Step 3: Data Cleaning
    df = data_cleaning(df_raw)
    
    # Step 4: Save Data & Print
    save_clean_data(df) 

if __name__ == '__main__':
    main()