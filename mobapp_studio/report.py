import pandas as pd
import matplotlib.pyplot as plt


def load_dataset(dataset):
    try:
        dataset = pd.read_csv(dataset)
        print(f"Dataset loaded successfully with {dataset.shape[0]} rows and {dataset.shape[1]} columns.")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None




def most_popular_apps(dataset, paid=True):
    # Get all categories in 'Category' column, return a list of them

    return


def most_popular_genres(dataset, by='Installs', paid=True):
    if paid:
        dataset = dataset[dataset['Type'] == 'Paid']
    genre_popularity = dataset.groupby('Genres')[by].sum().sort_values(ascending=False)
    return genre_popularity.head(10)


def total_installs_per_category_array(dataset, paid=True):
    if paid:
        dataset = dataset[dataset['Type'] == 'Paid']
    installs_per_category = dataset.groupby('Category')['Installs'].sum().sort_values(ascending=False)
    return installs_per_category


def total_installs_per_category_chart(dataset, paid=True):
    installs_per_category = total_installs_per_category_array(dataset, paid)
    plt.figure(figsize=(10, 6))
    installs_per_category.plot(kind='bar', color='skyblue')
    plt.title('Total Installs per Category')
    plt.xlabel('Category')
    plt.ylabel('Total Installs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def mean_price_per_category(dataset, paid=True):
    if paid:
        dataset = dataset[dataset['Type'] == 'Paid']
    mean_price = dataset.groupby('Category')['Price'].mean().sort_values(ascending=False)
    return mean_price


def most_expensive_apps_per_category(dataset, paid=True):
    if paid:
        dataset = dataset[dataset['Type'] == 'Paid']
    most_expensive_apps = dataset.groupby('Category').apply(lambda x: x.loc[x['Price'].idxmax()])[['App', 'Price']]
    return most_expensive_apps


def main():
    csv_files = ['archive/googleplaystore.csv', 'archive/googleplaystore_user_reviews.csv']
    df = load_dataset(csv_files[0])
    
    most_popular_apps(df, True)