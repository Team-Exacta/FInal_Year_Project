import pandas as pd
import json
import os

def main():
    print("Loading data...")
    try:
        # Reference aggregated_cleaned_reviews.csv from feature_extraction_outputs folder
        reviews_df = pd.read_csv('../feature_extraction_outputs/aggregated_cleaned_reviews.csv')
        triples_df = pd.read_csv('review_activity_triples.csv')
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure 'review_activity_triples.csv' exists in this folder and '../feature_extraction_outputs/aggregated_cleaned_reviews.csv' is accessible.")
        return

    print("Calculating total reviews per place...")
    # Count total reviews per place from the raw reviews dataset
    total_reviews_per_place = reviews_df['place_name'].value_counts().reset_index()
    total_reviews_per_place.columns = ['place_name', 'total_reviews']

    print("Aggregating activity mentions...")
    # Filter triples to Place -> Activity
    activity_triples = triples_df[(triples_df['subject_type'] == 'Place') & (triples_df['object_type'] == 'Activity')].copy()
    activity_triples['place_name'] = activity_triples['subject']
    activity_triples['activity'] = activity_triples['object']

    # Group by place and activity to collect unique evidence_ids (review IDs)
    activity_evidence = activity_triples.groupby(['place_name', 'activity']).agg(
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

    activity_evidence['dominant_sentiment'] = activity_evidence.apply(get_dominant_sentiment, axis=1)

    # Count unique reviews that mentioned the activity
    activity_evidence['review_count'] = activity_evidence['unique_reviews'].apply(len)

    # Merge to get total reviews for calculating percentage
    activity_evidence = pd.merge(activity_evidence, total_reviews_per_place, on='place_name', how='left')

    # Drop any places that might not exist in the reviews_df
    activity_evidence = activity_evidence.dropna(subset=['total_reviews'])

    # Calculate proportion of reviews mentioning the activity
    activity_evidence['review_percentage'] = (activity_evidence['review_count'] / activity_evidence['total_reviews']) * 100

    # Filter popular activities (>= 10% of reviews AND at least 5 reviews)
    THRESHOLD_PERCENT = 10.0
    MIN_REVIEWS = 5
    popular_activities = activity_evidence[(activity_evidence['review_percentage'] >= THRESHOLD_PERCENT) & 
                                         (activity_evidence['review_count'] >= MIN_REVIEWS)].copy()

    # Sort by place and percentage (descending)
    popular_activities = popular_activities.sort_values(by=['place_name', 'review_percentage'], ascending=[True, False])

    print(f"Saving popular_activities_by_place.csv...")
    csv_columns = ['place_name', 'activity', 'review_percentage', 'review_count', 'total_reviews', 
                   'avg_confidence', 'dominant_sentiment']
    
    # Format decimals for cleaner CSV output
    popular_activities['review_percentage'] = popular_activities['review_percentage'].round(2)
    popular_activities['avg_confidence'] = popular_activities['avg_confidence'].round(4)
    
    popular_activities[csv_columns].to_csv('popular_activities_by_place.csv', index=False)

    print("Saving activity_evidence_mapping.json...")
    mapping = {}
    for _, row in popular_activities.iterrows():
        place = row['place_name']
        activity = row['activity']
        evidence_list = row['unique_reviews']
        
        if place not in mapping:
            mapping[place] = {}
            
        mapping[place][activity] = {
            "review_percentage": row['review_percentage'],
            "review_count": row['review_count'],
            "avg_confidence": row['avg_confidence'],
            "dominant_sentiment": row['dominant_sentiment'],
            "evidence_ids": evidence_list
        }

    with open('activity_evidence_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=4)

    print(f"Done! Extracted {len(popular_activities)} popular activities across {popular_activities['place_name'].nunique()} places.")

if __name__ == "__main__":
    main()
