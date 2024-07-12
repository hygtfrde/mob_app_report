import os
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


report_dir = 'report_plots'
if not os.path.exists(report_dir):
    os.makedirs(report_dir)


def most_popular_apps(dataset, paid=True):
    if paid:
        filtered_data = dataset[dataset['Type'] == 'Paid']
    else:
        filtered_data = dataset[dataset['Type'] == 'Free']
    filtered_data = filtered_data.dropna(subset=['Rating'])
    idx_max_ratings = filtered_data.groupby('Category')['Rating'].idxmax()
    popular_apps = filtered_data.loc[idx_max_ratings]
    for _, row in popular_apps.iterrows():
        print(f"Category: {row['Category']}, Highest Rated App: {row['App']}, Rating: {row['Rating']}")
    
    return popular_apps


def most_popular_genres(dataset, paid=True):
    if paid:
        filtered_data = dataset[dataset['Type'] == 'Paid']
    else:
        filtered_data = dataset[dataset['Type'] == 'Free']
    filtered_data = filtered_data.dropna(subset=['Installs'])
    filtered_data['Installs'] = filtered_data['Installs'].str.replace('[^\d]', '', regex=True).astype(int)
    installs_per_genre = filtered_data.groupby('Genres')['Installs'].sum().sort_values(ascending=False)
    installs_list = installs_per_genre.reset_index().values.tolist()
    genres_with_numbers = {f"{i+1}": genre for i, genre in enumerate(installs_per_genre.index)}
    
    plt.figure(figsize=(10, 7))
    plt.pie(installs_per_genre, labels=genres_with_numbers.keys(), autopct='%1.1f%%', startangle=140)
    plt.title('Total Installs per Genre')
    plt.axis('equal')
    legend_labels = [f"{num}: {genre}" for num, genre in genres_with_numbers.items()]
    plt.legend(legend_labels, title="Genres", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    
    for label in legend_labels:
        print(label)
    
    return installs_list


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


# MAIN
def main():
    csv_files = ['archive/googleplaystore.csv', 'archive/googleplaystore_user_reviews.csv']
    df = load_dataset(csv_files[0])
    
    most_popular_apps(df, True)
    
    most_popular_genres(df, True)


if __name__ == '__main__':
    main()