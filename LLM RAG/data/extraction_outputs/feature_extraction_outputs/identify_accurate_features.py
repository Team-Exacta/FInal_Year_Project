import pandas as pd
import math

# Load the summary data
df = pd.read_csv('feature_summary_by_place.csv')

# 1. Filter out low confidence and low frequency noise
# Mentions >= 5 and Average Confidence >= 0.75
filtered_df = df[(df['total_mentions'] >= 5) & (df['average_confidence'] >= 0.75)].copy()

# 2. Calculate TF-IDF like score to find "defining" features per place
# TF: total_mentions of feature f in place p / total_mentions of all features in place p
place_totals = filtered_df.groupby('place_name')['total_mentions'].sum().reset_index(name='place_total_mentions')
filtered_df = pd.merge(filtered_df, place_totals, on='place_name')
filtered_df['TF'] = filtered_df['total_mentions'] / filtered_df['place_total_mentions']

# IDF: log(total number of places / number of places that have feature f)
total_places = filtered_df['place_name'].nunique()
feature_df_counts = filtered_df.groupby('feature')['place_name'].nunique().reset_index(name='doc_freq')
filtered_df = pd.merge(filtered_df, feature_df_counts, on='feature')

# Use math.log for natural logarithm
filtered_df['IDF'] = filtered_df['doc_freq'].apply(lambda x: math.log(total_places / x))

# TF-IDF
filtered_df['TF_IDF'] = filtered_df['TF'] * filtered_df['IDF']

# Also calculate a composite score: TF-IDF * average_confidence
# This balances uniqueness with the model's extraction confidence
filtered_df['defining_score'] = filtered_df['TF_IDF'] * filtered_df['average_confidence']

# Sort by place and defining score descending
filtered_df = filtered_df.sort_values(by=['place_name', 'defining_score'], ascending=[True, False])

# Get the top 5 most defining features per place
top_features_per_place = filtered_df.groupby('place_name').head(5)

# Save the detailed accurate features
filtered_df.to_csv('highly_accurate_features.csv', index=False)

# Save the top defining features
top_features_per_place.to_csv('top_defining_features_per_place.csv', index=False)

print(f"Total places processed: {total_places}")
print("Sample of top defining features for 5 places:")
for place in top_features_per_place['place_name'].unique()[:5]:
    print(f"\n{place}:")
    features = top_features_per_place[top_features_per_place['place_name'] == place]
    for _, row in features.iterrows():
        print(f"  - {row['feature']} (Score: {row['defining_score']:.4f}, Mentions: {row['total_mentions']}, Sentiment: {row['dominant_sentiment']})")

print("\nFiles 'highly_accurate_features.csv' and 'top_defining_features_per_place.csv' created.")
