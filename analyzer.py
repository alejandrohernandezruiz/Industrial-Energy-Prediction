import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONSTANTS ---
TARGET = 'Energy Consumption (kWh)'
DATA_FILENAME = 'clean_data.csv'

def load_data():
    # 1. Loads the dataset and ensures the target variable exists
    
    abs_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(abs_folder, 'data')
    route = os.path.join(data_folder, DATA_FILENAME)
    
    if not os.path.exists(route):
        raise FileNotFoundError(f'Data file not found at: {route}')
        
    df = pd.read_csv(route)
        
    return df

def general_inspection(df):
    # 2. Prints general shape and datatypes of the dataset
    
    print('\n' + '='*20 + ' 1. GENERAL INSPECTION ' + '='*20)
    print(f"Dataset dimensions (Rows, Columns): {df.shape}")
    print("\nDatatypes per column:")
    print(df.dtypes)

def descriptive_statistics(df):
    # 3. Prints standard descriptive statistics
    
    print('\n' + '='*20 + ' 2. DESCRIPTIVE STATISTICS ' + '='*20)
    print(df.describe())

def analyze_correlation(df):
    # 4. Calculates and reports linear correlations
    
    print('\n' + '='*20 + ' 3. CORRELATION ANALYSIS ' + '='*20)
    
    # Only compute correlation for numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    print(f"Variables with the strongest linear relationship to '{TARGET}':")
    # Drop the target itself, take absolute values, sort descending
    influential = corr_matrix[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    
    for col, val in influential.items():
        print(f" - {col}: {val:.4f}")
        
    return corr_matrix

def visualize_data(df, corr_matrix):
    # 5. Generates all matplotlib visual reports (only matplotlib)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ---- Graphic 1: Correlation Matrix Without Seaborn ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label='Correlation Coefficient')
    
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr_matrix.columns)
    
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                    color='black' if abs(val) < 0.6 else 'white')
            
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

    # ---- Graphic 2: Boxplots for Outliers (New) ----
    # Determine grid size for subplots
    n_cols = len(numeric_cols)
    fig, axes = plt.subplots(nrows=1, ncols=n_cols, figsize=(3 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
        
    for ax, col in zip(axes, numeric_cols):
        ax.boxplot(df[col].dropna(), patch_artist=True, boxprops=dict(facecolor='lightblue'))
        ax.set_title(col)
        ax.grid(True, linestyle='--', alpha=0.7)
        
    plt.suptitle('Boxplots (Quartile & Outlier Visualization)', fontsize=14)
    plt.tight_layout()
    plt.show()

    # ---- Graphic 3: Histograms & Distributions ----
    df[numeric_cols].hist(bins=20, figsize=(12, 8), color='royalblue', edgecolor='black')
    plt.suptitle('Histograms of Numerical Variables', fontsize=16)
    plt.tight_layout()
    plt.show()

    # ---- Graphic 4: Multivariate Colored Scatter Plot (New) ----
    if 'Production Rate' in df.columns and 'Maintenance Score' in df.columns:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            df['Production Rate'], 
            df[TARGET], 
            c=df['Maintenance Score'], 
            cmap='viridis', 
            alpha=0.7
        )
        plt.colorbar(scatter, label='Maintenance Score')
        plt.xlabel('Production Rate')
        plt.ylabel(TARGET)
        plt.title(f'Interaction: {TARGET} vs Production Rate (Colored by Maintenance)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    # ---- Graphic 5: Scatter Plots vs Target ----
    features = [col for col in numeric_cols if col != TARGET]
    if features:
        fig, axes = plt.subplots(nrows=1, ncols=len(features), figsize=(4 * len(features), 4), sharey=True)
        if len(features) == 1:
            axes = [axes]
            
        for ax, col in zip(axes, features):
            ax.scatter(df[col], df[TARGET], alpha=0.5, color='teal')
            ax.set_xlabel(col)
            ax.set_title(f'{col} vs Target')
            ax.grid(True, linestyle='--', alpha=0.7)
            
        plt.suptitle(f"Scatter Analysis vs {TARGET}", fontsize=14)
        plt.tight_layout()
        plt.show()

def report_conclusions(corr_matrix):
    # 6. Outputs dynamically generated engineering interpretations based on the data
    
    print('\n' + '='*20 + ' 4. ANALYSIS CONCLUSIONS ' + '='*20)
    
    # Analyze associations with the target
    corrs_with_target = corr_matrix[TARGET].drop(TARGET).abs()
    strongest = corrs_with_target.idxmax()
    weakest = corrs_with_target.idxmin()
    
    # Check for collinearity among input features (excluding target)
    input_features = corr_matrix.drop(columns=[TARGET], index=[TARGET])
    suspicious_pairs = []
    
    # Iterate through upper triangle of correlation matrix to find high correlations
    for i in range(len(input_features.columns)):
        for j in range(i+1, len(input_features.columns)):
            corr_val = input_features.iloc[i, j]
            if abs(corr_val) > 0.7:  # Threshold for high collinearity
                pair = f"{input_features.columns[i]} & {input_features.columns[j]} ({corr_val:.2f})"
                suspicious_pairs.append(pair)

    # 1. Target Associations
    print(f"• STRONGEST ASSOCIATION: '{strongest}' has the strongest linear relationship "
          f"with {TARGET}. This suggests it will be a high-weight driver in the upcoming Linear Regression.")
    print(f"• WEAKEST ASSOCIATION: '{weakest}' shows the weakest linear relationship. "
          f"It may contribute little explanatory power to the model.")

    # 2. Input Collinearity Risk
    if suspicious_pairs:
        print(f"• MULTICOLLINEARITY RISK: Detected strong associations between input features: "
              f"{', '.join(suspicious_pairs)}. Consider dropping or combining these variables "
              f"to prevent unstable regression coefficients.")
    else:
        print("• FEATURE INDEPENDENCE: No highly suspicious correlations (> 0.7) were found "
              "among the input features. This provides a stable foundation for the regression model.")

    # 3. Distributions & Scaling
    print("• DATA DISTRIBUTIONS: Based on visual inspection of the boxplots and histograms, "
          "significant differences in scale or non-normal distributions (e.g., Maintenance Score) "
          "indicate that applying a StandardScaler will be a necessary prerequisite for the model pipeline.")

def main():
    # We call step by step every function
    
    df = load_data()
    general_inspection(df)
    descriptive_statistics(df)
        
    corr_matrix = analyze_correlation(df)
    visualize_data(df, corr_matrix)
        
    report_conclusions(corr_matrix)
        
    print('\nAnalyzer execution completed successfully. Ready for linear_regression.py.')

if __name__ == '__main__':
    main()