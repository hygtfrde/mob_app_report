import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import os


def load_dataset(dataset):
    try:
        dataset = pd.read_csv(dataset)
        print(f"Dataset loaded successfully with {dataset.shape[0]} rows and {dataset.shape[1]} columns.")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def print_summarize_dataset(dataset):
    print("Dataset Summary: ", dataset.describe())
    print("Dataset Head: ", dataset.head(20))


def clean_dataset(dataset):
    print("Pre-Cleaning Number of Rows: ", len(dataset))
    dataset_cleaned = dataset.drop_duplicates()
    print(f"Removed {dataset.shape[0] - dataset_cleaned.shape[0]} duplicate rows.")
    missing_values = dataset_cleaned.isnull().sum()
    print("Missing values per column: ", missing_values[missing_values > 0])
    
    # Optional: drop rows with missing values or unwanted values such as NaN
    dataset_cleaned = dataset_cleaned.dropna()
    print(f"Removed rows with missing values, new shape: {dataset_cleaned.shape}")
    
    # Fix numerical column data types
    dataset_cleaned['Reviews'] = pd.to_numeric(dataset_cleaned['Reviews'], errors='coerce')
    
    dataset_cleaned['Size'] = dataset_cleaned['Size'].str.rstrip('M')
    dataset_cleaned['Size'] = pd.to_numeric(dataset_cleaned['Size'], errors='coerce')
    dataset_cleaned['Size'] = dataset_cleaned['Size'] * 1000000
    
    dataset_cleaned['Installs'] = dataset_cleaned['Installs'].str.replace('[^\d]', '', regex=True)
    dataset_cleaned['Installs'] = pd.to_numeric(dataset_cleaned['Installs'], errors='coerce')

    dataset_cleaned['Price'] = dataset_cleaned['Price'].str.replace('[^\d.]', '', regex=True)
    dataset_cleaned['Price'] = pd.to_numeric(dataset_cleaned['Price'], errors='coerce')
    
    
    # Detect outliers 
    def detect_outliers_z_score(column_data, threshold=1.5):
        z_scores = np.abs(stats.zscore(column_data))
        outliers_count = np.sum(z_scores > threshold)
        return outliers_count

    columns_to_check = ['Rating', 'Reviews', 'Size', 'Installs', 'Price']
    for column in columns_to_check:
        outliers_in_df = detect_outliers_z_score(dataset_cleaned[column])
        print(f'Outliers in {column}: ', outliers_in_df)

    print("Post-Cleaning Number of Rows: ", len(dataset_cleaned))
    return dataset_cleaned



def print_histograms(dataset):
    numeric_columns = dataset.select_dtypes(include=['float64', 'int64']).columns

    hist_dir = 'histograms'
    if not os.path.exists(hist_dir):
        os.makedirs(hist_dir)

    for column in numeric_columns:
        plt.figure(figsize=(7, 5))
        counts, bins, patches = plt.hist(dataset[column], bins=30, edgecolor='black')

        plt.title(column + " Frequency")
        plt.xlabel(column)
        plt.ylabel('Frequency')
        x_min = dataset[column].min()
        x_max = dataset[column].max()
        plt.xlim(x_min, x_max)

        # Adding arrows or text labels for non-negative values
        for count, bin_edge in zip(counts, bins):
            if count > 0 and count < max(counts) * 0.05:  # Threshold for small bars
                plt.annotate(f'{int(count)}', 
                             xy=(bin_edge, count), 
                             xytext=(bin_edge, count + max(counts) * 0.02),
                             arrowprops=dict(facecolor='red', shrink=0.05))

        plt.tight_layout()
        plt.savefig(os.path.join(hist_dir, f'{column}_histogram.png'))
        plt.show()
        plt.close()


def compute_correlations_matrix(dataset):
    corr_dir = 'correlations'
    if not os.path.exists(corr_dir):
        os.makedirs(corr_dir)
        
    numeric_dataset = dataset.select_dtypes(include=['number'])
    
    # Remove columns with constant values
    numeric_dataset = numeric_dataset.loc[:, (numeric_dataset != numeric_dataset.iloc[0]).any()]

    # Check for missing values and drop them or fill them
    if numeric_dataset.isnull().values.any():
        print("Warning: Dataset contains NaNs, these will be dropped for correlation calculation.")
        numeric_dataset = numeric_dataset.dropna(axis=1, how='any')
    
    correlation_matrix = numeric_dataset.corr()
    print("Correlation Matrix:")
    print(correlation_matrix)
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    
    # Ensure the matrix is not empty
    if not correlation_matrix.empty:
        mask = np.zeros_like(correlation_matrix)
        mask[np.triu_indices_from(mask)] = True
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, mask=mask, vmin=-1, vmax=1, linewidths=0.5, square=True, cbar_kws={"shrink": .5})
        plt.title('Correlation Matrix Heatmap')
    else:
        print("Correlation matrix is empty or all values are 1.")
    plt.savefig(os.path.join(corr_dir, f'correlation_matrix.png'))
    plt.show()
    plt.close()
    return correlation_matrix


def print_scatter_matrix(dataset, height=2.5):
    scatter_dir = 'scatter'
    if not os.path.exists(scatter_dir):
        os.makedirs(scatter_dir)

    sns.pairplot(dataset, height=height, diag_kind='hist')
    plt.savefig(os.path.join(scatter_dir, f'scatter_matrix.png'))
    plt.show()
    plt.close()



def analyze_user_reviews(file_path):
    analytics_dir = 'user_analytics'
    if not os.path.exists(analytics_dir):
        os.makedirs(analytics_dir)
        
    df = pd.read_csv(file_path)
    
    df = df.dropna(subset=['Sentiment'])
    
    df['Sentiment_Polarity'] = pd.to_numeric(df['Sentiment_Polarity'], errors='coerce')
    df['Sentiment_Subjectivity'] = pd.to_numeric(df['Sentiment_Subjectivity'], errors='coerce')
    
    user_reviews_df = df.groupby('App').agg({
        'Sentiment': lambda x: x.value_counts(normalize=True).to_dict(),
        'Sentiment_Polarity': ['mean', 'min', 'max'],
        'Sentiment_Subjectivity': ['mean', 'min', 'max']
    }).reset_index()
    
    # Rename columns for clarity
    user_reviews_df.columns = ['App', 'Sentiment_Distribution', 'Polarity_Mean', 'Polarity_Min', 'Polarity_Max', 'Subjectivity_Mean', 'Subjectivity_Min', 'Subjectivity_Max']
    user_reviews_df.to_parquet(f'{analytics_dir}/user_review_analytics.parquet', index=False)
    
    
    
    # Sort apps by mean polarity and subjectivity
    top_10_apps_polarity = user_reviews_df.nlargest(10, 'Polarity_Mean')
    lowest_10_apps_polarity = user_reviews_df.nsmallest(10, 'Polarity_Mean')
    top_10_apps_subjectivity = user_reviews_df.nlargest(10, 'Subjectivity_Mean')
    lowest_10_apps_subjectivity = user_reviews_df.nsmallest(10, 'Subjectivity_Mean')
    
    # Merge the DataFrames as required
    # 1) High polarity and low subjectivity
    high_polarity_low_subjectivity = pd.concat(
        [top_10_apps_polarity.drop(columns='Sentiment_Distribution'), 
         lowest_10_apps_subjectivity.drop(columns='Sentiment_Distribution')]
    ).drop_duplicates(subset=['App'])

    # Filter out apps with non-positive polarity in high polarity table
    high_polarity_low_subjectivity = high_polarity_low_subjectivity[high_polarity_low_subjectivity['Polarity_Mean'] > 0]

    # 2) Low polarity and high subjectivity
    low_polarity_high_subjectivity = pd.concat(
        [lowest_10_apps_polarity.drop(columns='Sentiment_Distribution'), 
         top_10_apps_subjectivity.drop(columns='Sentiment_Distribution')]
    ).drop_duplicates(subset=['App'])

    # Filter out apps with non-negative polarity in low polarity table
    low_polarity_high_subjectivity = low_polarity_high_subjectivity[low_polarity_high_subjectivity['Polarity_Mean'] < 0]

    # Save the merged DataFrames to Excel files
    high_polarity_low_subjectivity.to_excel(f'{analytics_dir}/high_polarity_low_subjectivity.xlsx', index=False)
    low_polarity_high_subjectivity.to_excel(f'{analytics_dir}/low_polarity_high_subjectivity.xlsx', index=False)
    
    
    return user_reviews_df



def check_missing_values(df, columns_to_check, check_all=False):
    if check_all:
        missing_columns = [col for col in df.columns if df[col].isnull().any()]
        if missing_columns:
            print(f"YES MISSING VALUES in columns: {', '.join(missing_columns)}")
        else:
            print("No missing values in any column.")
    else:
        if not columns_to_check:
            print("Error: Empty list of columns to check.")
            return
        missing_columns = [col for col in columns_to_check if df[col].isnull().any()]
        if missing_columns:
            print(f"YES MISSING VALUES in columns: {', '.join(missing_columns)}")
        else:
            print("No missing values in specified columns.")


# -------------------------------
# MAIN
def main():
    csv_files = ['archive/googleplaystore.csv', 'archive/googleplaystore_user_reviews.csv']
    df = load_dataset(csv_files[0])
    df_dir = 'dataframe_output'
    if not os.path.exists(df_dir):
        os.makedirs(df_dir)
    
    if df is not None:
        print_summarize_dataset(df)
        
        cleaned_df = clean_dataset(df)
        cleaned_df.to_excel(f'{df_dir}/cleaned_df.xlsx', index=False)
        check_missing_values(cleaned_df, [], True)
        
        print_histograms(cleaned_df)
        
        correlations = compute_correlations_matrix(cleaned_df)
        print(f'Correlations Matrix\n{correlations}')
        
        print_scatter_matrix(cleaned_df)
        
    user_review_analytics = analyze_user_reviews(csv_files[1])
    print(f"user_review_analytics\n{user_review_analytics}")


# MAIN  
if __name__ == '__main__':
    main()