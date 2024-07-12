import os
import openpyxl
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
    total_installs = installs_per_genre.sum()
    percentages = (installs_per_genre / total_installs) * 100
    threshold = 5
    main_genres = percentages[percentages >= threshold]
    other_genres = percentages[percentages < threshold]
    main_installs = installs_per_genre[main_genres.index]
    other_installs = installs_per_genre[other_genres.index].sum()
    if other_installs > 0:
        main_installs['Other'] = other_installs
        main_genres['Other'] = (other_installs / total_installs) * 100
    labels = main_genres.index
    sizes = main_installs.values
    plt.figure(figsize=(10, 7))
    def autopct_generator(pct):
        return '%1.1f%%' % pct if pct >= threshold else ''
    wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct=autopct_generator, startangle=140)
    for autotext in autotexts:
        autotext.set_color('white')

    plt.title('Total Installs per Genre')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'{report_dir}/total_installs_per_genre.png', bbox_inches='tight')
    plt.show()
    
    installs_list = installs_per_genre.reset_index().values.tolist()
    main_installs_list = main_installs.reset_index().values.tolist()
    output_df = pd.DataFrame(main_installs_list, columns=['Genre', 'Installs'])
    output_df.to_excel(f'{report_dir}/installs_per_genre.xlsx', index=False)

    for label in labels:
        print(label)

    return installs_list



def total_installs_per_category_array(dataset, paid=True):
    if paid:
        filtered_data = dataset[dataset['Type'] == 'Paid']
    else:
        filtered_data = dataset[dataset['Type'] == 'Free']
        
    filtered_data = filtered_data.dropna(subset=['Installs'])
    filtered_data['Installs'] = filtered_data['Installs'].str.replace('[^\d]', '', regex=True).astype(int)
    
    installs_per_category = filtered_data.groupby('Category')['Installs'].sum().sort_values(ascending=False)
    installs_list = installs_per_category.reset_index().values.tolist()
    
    output_df = pd.DataFrame(installs_list, columns=['Category', 'Total Installs'])
    output_df.to_excel(f'{report_dir}/total_installs_per_category.xlsx', index=False)
    
    return installs_list


def total_installs_per_category_chart(dataset, paid=True):
    installs_per_category = total_installs_per_category_array(dataset, paid)
    categories = [item[0] for item in installs_per_category]
    total_installs = [item[1] for item in installs_per_category]
    plt.figure(figsize=(10, 6))
    plt.bar(categories, total_installs, color='skyblue')
    plt.title('Total Installs per Category')
    plt.xlabel('Category')
    plt.ylabel('Total Installs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{report_dir}/total_installs_per_category_chart.png')
    plt.show()


def clean_price(price):
    try:
        return float(price.strip('$'))  # Assuming prices are formatted as '$X.XX'
    except ValueError:
        return None


def mean_price_per_category(dataset, paid=True):
    if paid:
        dataset = dataset[dataset['Type'] == 'Paid']
    else:
        dataset = dataset[dataset['Type'] == 'Free']
    
    # Clean up 'Price' column
    dataset['Price'] = dataset['Price'].apply(clean_price)
    
    # Filter out rows where 'Price' couldn't be converted to numeric
    dataset = dataset.dropna(subset=['Price'])
    
    # Group by 'Category' and calculate mean price
    mean_price = dataset.groupby('Category')['Price'].mean().sort_values(ascending=False)
    return mean_price

def mean_price_per_category_chart(dataset, paid=True):
    mean_prices = mean_price_per_category(dataset, paid)
    categories = mean_prices.index
    prices = mean_prices.values
    
    plt.figure(figsize=(10, 6))
    plt.bar(categories, prices, color='skyblue')
    plt.title('Mean Price per Category')
    plt.xlabel('Category')
    plt.ylabel('Mean Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{report_dir}/mean_price_per_category_chart.png')
    plt.show()


def most_expensive_apps_per_category(dataset, paid=True):
    if paid:
        dataset = dataset[dataset['Type'] == 'Paid']
    else:
        dataset = dataset[dataset['Type'] == 'Free']
    dataset['Price'] = dataset['Price'].apply(clean_price)
    dataset = dataset.dropna(subset=['Price'])
    idx_max_prices = dataset.groupby('Category')['Price'].idxmax()
    most_expensive_apps = dataset.loc[idx_max_prices]
    most_expensive_apps.to_excel(f'{report_dir}/most_expensive_apps_per_category.xlsx', index=False)
    
    return most_expensive_apps


# MAIN
def main():
    csv_files = ['archive/googleplaystore.csv', 'archive/googleplaystore_user_reviews.csv']
    df = load_dataset(csv_files[0])
    
    # most_popular_apps(df, True)
    
    # most_popular_genres(df, True)
    
    # total_installs_per_category_array(df, True)
    
    # total_installs_per_category_chart(df, True)

    # mean_price_per_category_chart(df, True)
    
    most_expensive_apps_per_category(df, True)


if __name__ == '__main__':
    main()