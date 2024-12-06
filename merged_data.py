import pandas as pd

# Load datasets
data = pd.read_csv("./data/data.csv")
cities = pd.read_csv("./data/top_100_cities.csv")

# Optimize by selecting relevant columns and indexing
cities = cities[['city_name', 'latitude', 'longitude']]
cities.set_index('city_name', inplace=True)

# Convert columns to categorical for memory efficiency
data['from_city'] = data['from_city'].astype('category')
data['to_city'] = data['to_city'].astype('category')

# Perform the merge using indexed DataFrames
data = pd.merge(data, cities, left_on='from_city', right_index=True, suffixes=('', '_from'))
data = pd.merge(data, cities, left_on='to_city', right_index=True, suffixes=('', '_to'))

# Rename coordinate columns for clarity
data.rename(columns={'latitude_from': 'start_lat', 'longitude_from': 'start_lng',
                     'latitude': 'end_lat', 'longitude': 'end_lng'}, inplace=True)
