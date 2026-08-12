import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONSTANTS ---

TARGET = 'Energy Consumption (kWh)'
DATA_FILNAME = 'clean_data.csv'

TRUE_COEFFICIENTS = np.array([150.0, 12.5, 3.8, 45.0, -2.1, -0.5])

def load_data():
    # 1. Load Data Function
    
    abs_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(abs_folder, 'data')
    route = os.path.join(data_folder, DATA_FILNAME)

    if not os.path.exists(route):
        raise FileNotFoundError(f'Data file not found at: {route}')

    df = pd.read_csv(route)

    return df

def separate_xy(df):
    # 2. Separate X and y Function
    
    y_true = df[TARGET].values.reshape(-1, 1)

    x_features = df[
        [
            'Production Rate',
            'Furnace Temperature',
            'Machine Operation Hours',
            'Ambient Temperature',
            'Maintenance Score'
        ]
    ].values

    return y_true, x_features

def design_matrix(x_features):
    # 3. Design Matrix Function (X)
    
    m = len(x_features)

    # Add a column of 1s for the intercept/bias term
    bias_column = np.ones((m, 1))
    X = np.hstack((bias_column, x_features))

    return X


def train_test_split(X, y, train_size=0.8, random_seed=42):
    # 4. Train Test Split Function (80% train and 20% test)
    
    np.random.seed(random_seed)

    m = len(X)
    indices = np.arange(m)

    np.random.shuffle(indices)

    train_limit = int(m * train_size)

    train_idx = indices[:train_limit]
    test_idx = indices[train_limit:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test


def normal_equation(X_train, y_train):
    # 5. Find the optimal Beta Vector by solving the Normal Equation
    # BETA = (X^T * X)^-1 * X^T * y

    X_transpose = X_train.T

    beta_hat = (np.linalg.inv(X_transpose @ X_train) @ X_transpose @ y_train)

    return beta_hat


def display_coefficients(beta_hat):
    # 6. Display and Compare coefficients of our parameter vector (beta hat)
    
    print('\n' + '=' * 20 + ' 1. BETA VECTOR - COEFFICIENTS COMPARISON ' + '=' * 20 + '\n')

    names = [
        'Intercept (Bias)',
        'Production Rate (x1)',
        'Furnace Temperature (x2)',
        'Machine Operation Hours (x3)',
        'Ambient Temperature (x4)',
        'Maintenance Score (x5)'
    ]

    for i in range(len(beta_hat)):
        variable_name = names[i]

        calculated_value = beta_hat[i].item()
        real_value = TRUE_COEFFICIENTS[i]

        error = real_value - calculated_value

        print(f"Variable: {variable_name}")
        print(f"  -> Retrieved by the model: {calculated_value:.6f}")
        print(f"  -> Original actual value:  {real_value:.6f}")
        print(f"  -> Difference / Error:      {error:.6f}")
        print("-" * 40)


def make_predictions(X_train, X_test, beta_hat):
    # 7. Make Predictions with the Matrix Equation: y = X * Beta

    y_pred_train = X_train @ beta_hat
    y_pred_test = X_test @ beta_hat

    return y_pred_train, y_pred_test


def compute_metrics(y_true, y_pred, dataset_label='Dataset'):
    # 8. Compute MSE, RMSE and R² manually

    # MSE = 1/n * sum((y_true - y_pred)^2)
    residuals = y_true - y_pred
    mse = np.mean(residuals ** 2)

    # RMSE = sqrt(MSE)
    rmse = np.sqrt(mse)

    # R² = 1 - (SS_res / SS_tot)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        r_squared = 0.0
    else:
        r_squared = 1 - (ss_res / ss_tot)

    print(f"\nMetrics for {dataset_label}:")
    print(f"  -> Mean Squared Error (MSE): {mse:.4f}")
    print(f"  -> Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  -> R² Score: {r_squared:.4f}")

    return mse, rmse, r_squared


def plot_regression_analysis(y_train, y_pred_train, y_test, y_pred_test):
    # 9. Use Matplotlib to plot our train vs test and errors
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    # --- PLOT 1: Real vs. Predicted ---
    # Train Data in Teal and Test Data in Orange
    ax1.scatter(
        y_train,
        y_pred_train,
        alpha=0.5,
        color='teal',
        label='Train Data'
    )

    ax1.scatter(
        y_test,
        y_pred_test,
        alpha=0.5,
        color='darkorange',
        label='Test Data'
    )

    # Ideal Reference Line: where predicted perfectly equals real data (y = x)
    all_y = np.concatenate([y_train, y_test])

    ideal_min = all_y.min()
    ideal_max = all_y.max()

    ax1.plot(
        [ideal_min, ideal_max],
        [ideal_min, ideal_max],
        color='red',
        linestyle='--',
        linewidth=2,
        label='Perfect Prediction'
    )

    ax1.set_xlabel('Real Energy Consumption (kWh)')
    ax1.set_ylabel('Predicted Energy Consumption (kWh)')
    ax1.set_title('Real vs. Predicted Values')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- PLOT 2: Residual Plot ---
    # Residuals = Real Value - Predicted Value
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test

    ax2.scatter(
        y_pred_train,
        residuals_train,
        alpha=0.5,
        color='teal',
        label='Train Residuals'
    )

    ax2.scatter(
        y_pred_test,
        residuals_test,
        alpha=0.5,
        color='darkorange',
        label='Test Residuals'
    )

    # Zero Error Baseline: where error is exactly zero
    ax2.axhline(
        y=0,
        color='red',
        linestyle='--',
        linewidth=2
    )

    ax2.set_xlabel('Predicted Energy Consumption (kWh)')
    ax2.set_ylabel('Residuals (Real - Predicted)')
    ax2.set_title('Residuals Analysis (Error Distribution)')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle(
        'Linear Regression Model Evaluation',
        fontsize=16,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.show()


def predict_new_conditions(
    production_rate,
    furnace_temp,
    machine_hours,
    ambient_temp,
    maintenance_score,
    beta_hat
):
    # 10. Predict Energy Consumption Based On New Conditions
    print(
        "\n" + "=" * 20
        + " 3. PRODUCTION PREDICTION SYSTEM "
        + "=" * 20
        + "\n"
    )

    # Create the feature vector
    raw_features = np.array([
        production_rate,
        furnace_temp,
        machine_hours,
        ambient_temp,
        maintenance_score
    ])

    # Add the bias/intercept term
    features = np.insert(raw_features, 0, 1.0)

    # Convert the vector into a row of the Design Matrix
    design_row = features.reshape(1, -1)

    # Prediction: y = X * Beta
    predicted_energy = design_row @ beta_hat

    final_result = predicted_energy.item()

    print("Engineering Inputs Provided:")
    print(f"  -> Production Rate:        {production_rate}")
    print(f"  -> Furnace Temperature:    {furnace_temp} °C")
    print(f"  -> Machine Op. Hours:      {machine_hours} hrs")
    print(f"  -> Ambient Temperature:    {ambient_temp} °C")
    print(f"  -> Maintenance Score:      {maintenance_score}")

    print(
        f"\n   FORECASTED ENERGY CONSUMPTION: "
        f"{final_result:.2f} kWh"
    )

    return final_result


def main():
    print("=== STARTING MULTIPLE LINEAR REGRESSION MODEL PIPELINE ===")

    # Step 1: Load Data
    df = load_data()

    # Step 2: Separate X and y
    y_true, x_features = separate_xy(df)

    # Step 3: Build Design Matrix
    X = design_matrix(x_features)

    # Step 4: Train & Test Manual Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_true,
        train_size=0.8,
        random_seed=42
    )

    # Step 5: Solve Normal Equation (using ONLY training data)
    beta_hat = normal_equation(X_train, y_train)

    # Step 6: Display Coefficients Comparison
    display_coefficients(beta_hat)

    # Step 7: Make Predictions
    y_pred_train, y_pred_test = make_predictions(
        X_train,
        X_test,
        beta_hat
    )

    # Step 8: Compute Metrics
    print(
        '\n' + '=' * 20
        + ' 2. EVALUATION METRICS '
        + '=' * 20
    )

    compute_metrics(
        y_train,
        y_pred_train,
        'Training Set (In-Sample)'
    )

    compute_metrics(
        y_test,
        y_pred_test,
        'Test Set (Out-of-Sample)'
    )

    # Step 9: Plot Visual Diagnostics
    plot_regression_analysis(
        y_train,
        y_pred_train,
        y_test,
        y_pred_test
    )

    # Step 10: Simulate Production Call with an example scenario
    # Let's test with: 140 rate, 1100°C temp, 18 hrs, 22°C ambient, 85 maintenance
    predict_new_conditions(
        140.0,
        1100.0,
        18.0,
        22.0,
        85.0,
        beta_hat
    )

    print(
        "\n=== PIPELINE EXECUTION COMPLETED SUCCESSFULLY ==="
    )


if __name__ == '__main__':
    main()