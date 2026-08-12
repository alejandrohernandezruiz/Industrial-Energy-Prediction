import numpy as np
import pandas as pd
import os

# Route Configuration

abs_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(abs_folder, 'data')

os.makedirs(data_folder, exist_ok=True)

route = os.path.join(data_folder, 'raw_data.csv')

# Constants

N_FEATURES = 5
M_SAMPLES = 1000

MEAN_1, STD_1 = 150, 8.23   # Production Rate
MEAN_2, STD_2 = 800, 11.1   # Furnace Temperature
MEAN_3, STD_3 = 18, 0.95    # Machine Operation Hours
MEAN_4, STD_4 = 25, 6.0     # Ambient Temperature
LOW_5, HIGH_5 = 0, 100      # Maintenance Score

FEATURE_COLUMNS = [
    'Production Rate',
    'Furnace Temperature',
    'Machine Operation Hours',
    'Ambient Temperature',
    'Maintenance Score'
]

TARGET = 'Energy Consumption (kWh)'

TRUE_COEFFICIENTS = np.array([150.0, 12.5, 3.8, 45.0, -2.1, -0.5])

# Functions

def generate_features():
    # 1. Generate Clean Data
    
    np.random.seed(42)

    x1 = np.random.normal(MEAN_1, STD_1, M_SAMPLES)
    x2 = np.random.normal(MEAN_2, STD_2, M_SAMPLES)
    x3 = np.random.normal(MEAN_3, STD_3, M_SAMPLES)
    x4 = np.random.normal(MEAN_4, STD_4, M_SAMPLES)
    x5 = np.random.uniform(LOW_5, HIGH_5, M_SAMPLES)

    return x1, x2, x3, x4, x5


def create_design_matrix(x1, x2, x3, x4, x5):
    # 2. Build the Design Matrix (X)

    bias_column = np.ones((M_SAMPLES, 1))
    x_features = np.column_stack((x1, x2, x3, x4, x5))

    X = np.column_stack((bias_column, x_features))

    return X


def generate_target(X):
    # 3. Generate Energy Consumption

    noise = np.random.normal(0, 40, M_SAMPLES)

    y = (X @ TRUE_COEFFICIENTS) + noise

    return y


def create_dataframe(x1, x2, x3, x4, x5, y):
    # 4. Create DataFrame

    data = {
        'Production Rate': x1,
        'Furnace Temperature': x2,
        'Machine Operation Hours': x3,
        'Ambient Temperature': x4,
        'Maintenance Score': x5,
        'Energy Consumption (kWh)': y
    }

    df = pd.DataFrame(data)

    return df


def inject_corruption(df):
    # 5. Inject Corruption for later Data Cleaning

    df = df.copy()

    # A. Missing values

    df.loc[df.sample(frac=0.05, random_state=42).index, 'Furnace Temperature'] = np.nan
    df.loc[df.sample(frac=0.03, random_state=42).index, 'Ambient Temperature'] = np.nan

    # B. Outliers

    n_outliers_prod = int(M_SAMPLES * 0.03)
    n_outliers_mach = int(M_SAMPLES * 0.02)

    outliers_prod = np.random.choice([0.0, 450.0], n_outliers_prod, True)

    outliers_mach = np.random.choice([99.0], n_outliers_mach, True)

    idx_prod = df.sample(n_outliers_prod, random_state=12).index

    df.loc[idx_prod, 'Production Rate'] = outliers_prod

    idx_mach = df.sample(n_outliers_mach, random_state=34).index

    df.loc[idx_mach, 'Machine Operation Hours'] = outliers_mach

    # C. Duplicate rows

    df_duplicates = df.iloc[0:200]

    df = pd.concat([df, df_duplicates], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def save_data(df):
    # 6. Save DataFrame

    df.to_csv(route, index=False)

    print('\nCSV file successfully generated:\n')
    print(df.head(10))


def main():
    print('--- STARTING DATA GENERATION PIPELINE ---')

    # Step 1: Generate features
    x1, x2, x3, x4, x5 = generate_features()

    # Step 2: Build Design Matrix
    X = create_design_matrix(x1, x2, x3, x4, x5)

    # Step 3: Generate target
    y = generate_target(X)

    # Step 4: Create DataFrame
    df = create_dataframe(x1, x2, x3, x4, x5, y)

    # Step 5: Inject corruption
    df = inject_corruption(df)

    # Step 6: Save data
    save_data(df)

    print('\n--- DATA GENERATION PIPELINE COMPLETED SUCCESSFULLY ---')


if __name__ == '__main__':
    main()