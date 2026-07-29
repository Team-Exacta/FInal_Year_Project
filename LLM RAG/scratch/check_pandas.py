import pandas as pd

df = pd.read_csv('data/graph/poi_data.csv')
for index, row in df.iterrows():
    if row['poi_name'] in ['Maha Saman Dewalaya', 'Kirinda Viharaya']:
        print(f"Name: {row['poi_name']}")
        print(f"Full Address: {row['full_address']}")
        print(f"City: {row['city']}")
        print(f"District: {row['district']}")
        print(f"Province: {row['province']}")
        print("---")
