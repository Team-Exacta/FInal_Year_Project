import pandas as pd
import json
import os

def main():
    print("Loading data...")
    try:
        reviews_df = pd.read_csv('aggregated_cleaned_reviews.csv')
        triples_df = pd.read_csv('review_feature_triples.csv')
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    print("Calculating total reviews per place...")
    # Count total reviews per place from the raw reviews dataset
    total_reviews_per_place = reviews_df['place_name'].value_counts().reset_index()
    total_reviews_per_place.columns = ['place_name', 'total_reviews']

    print("Aggregating feature mentions...")
    # Filter triples to Place -> Feature
    feature_triples = triples_df[(triples_df['subject_type'] == 'Place') & (triples_df['object_type'] == 'Feature')].copy()
    feature_triples['place_name'] = feature_triples['subject']
    feature_triples['feature'] = feature_triples['object']

    # Group by place and feature to collect unique evidence_ids (review IDs)
    feature_evidence = feature_triples.groupby(['place_name', 'feature']).agg(
        unique_reviews=('evidence_id', lambda x: sorted(list(set(x)))),
        avg_confidence=('confidence', 'mean'),
        positive_mentions=('sentiment', lambda x: (x == 'positive').sum()),
        negative_mentions=('sentiment', lambda x: (x == 'negative').sum()),
        neutral_mentions=('sentiment', lambda x: (x == 'neutral').sum())
    ).reset_index()

    # Determine dominant sentiment
    def get_dominant_sentiment(row):
        counts = {'positive': row['positive_mentions'], 'negative': row['negative_mentions'], 'neutral': row['neutral_mentions']}
        return max(counts, key=counts.get)

    feature_evidence['dominant_sentiment'] = feature_evidence.apply(get_dominant_sentiment, axis=1)

    # Count unique reviews that mentioned the feature
    feature_evidence['review_count'] = feature_evidence['unique_reviews'].apply(len)

    # Merge to get total reviews for calculating percentage
    feature_evidence = pd.merge(feature_evidence, total_reviews_per_place, on='place_name', how='left')

    # Drop any places that might not exist in the reviews_df
    feature_evidence = feature_evidence.dropna(subset=['total_reviews'])

    # Calculate proportion of reviews mentioning the feature
    feature_evidence['review_percentage'] = (feature_evidence['review_count'] / feature_evidence['total_reviews']) * 100

    # Filter popular features (>= 10% of reviews AND at least 5 reviews)
    THRESHOLD_PERCENT = 10.0
    MIN_REVIEWS = 5
    popular_features = feature_evidence[(feature_evidence['review_percentage'] >= THRESHOLD_PERCENT) & 
                                         (feature_evidence['review_count'] >= MIN_REVIEWS)].copy()

    # Sort by place and percentage (descending)
    popular_features = popular_features.sort_values(by=['place_name', 'review_percentage'], ascending=[True, False])

    print(f"Saving popular_features_by_place.csv...")
    csv_columns = ['place_name', 'feature', 'review_percentage', 'review_count', 'total_reviews', 
                   'avg_confidence', 'dominant_sentiment']
    
    # Format decimals for cleaner CSV output
    popular_features['review_percentage'] = popular_features['review_percentage'].round(2)
    popular_features['avg_confidence'] = popular_features['avg_confidence'].round(4)
    
    popular_features[csv_columns].to_csv('popular_features_by_place.csv', index=False)

    print("Saving feature_evidence_mapping.json...")
    mapping = {}
    for _, row in popular_features.iterrows():
        place = row['place_name']
        feature = row['feature']
        evidence_list = row['unique_reviews']
        
        if place not in mapping:
            mapping[place] = {}
            
        mapping[place][feature] = {
            "review_percentage": row['review_percentage'],
            "review_count": row['review_count'],
            "avg_confidence": row['avg_confidence'],
            "dominant_sentiment": row['dominant_sentiment'],
            "evidence_ids": evidence_list
        }

    with open('feature_evidence_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=4)

    print(f"Done! Extracted {len(popular_features)} popular features across {popular_features['place_name'].nunique()} places.")

if __name__ == "__main__":
    main()
