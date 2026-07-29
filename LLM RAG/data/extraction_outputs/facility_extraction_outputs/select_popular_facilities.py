import pandas as pd
import json
import os
import shutil

def main():
    print("Loading data...")
    try:
        # Reference aggregated_cleaned_reviews.csv from feature_extraction_outputs folder
        reviews_df = pd.read_csv('../feature_extraction_outputs/aggregated_cleaned_reviews.csv')
        triples_df = pd.read_csv('review_facility_triples.csv')
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure 'review_facility_triples.csv' exists in this folder and '../feature_extraction_outputs/aggregated_cleaned_reviews.csv' is accessible.")
        return

    print("Calculating total reviews per place...")
    # Count total reviews per place from the raw reviews dataset
    total_reviews_per_place = reviews_df['place_name'].value_counts().reset_index()
    total_reviews_per_place.columns = ['place_name', 'total_reviews']

    print("Aggregating facility mentions...")
    # Filter triples to Place -> Facility
    facility_triples = triples_df[(triples_df['subject_type'] == 'Place') & (triples_df['object_type'] == 'Facility')].copy()
    facility_triples['place_name'] = facility_triples['subject']
    facility_triples['facility'] = facility_triples['object']

    # Group by place and facility to collect unique evidence_ids (review IDs)
    facility_evidence = facility_triples.groupby(['place_name', 'facility']).agg(
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

    facility_evidence['dominant_sentiment'] = facility_evidence.apply(get_dominant_sentiment, axis=1)

    # Count unique reviews that mentioned the facility
    facility_evidence['review_count'] = facility_evidence['unique_reviews'].apply(len)

    # Merge to get total reviews for calculating percentage
    facility_evidence = pd.merge(facility_evidence, total_reviews_per_place, on='place_name', how='left')

    # Drop any places that might not exist in the reviews_df
    facility_evidence = facility_evidence.dropna(subset=['total_reviews'])

    # Calculate proportion of reviews mentioning the facility
    facility_evidence['review_percentage'] = (facility_evidence['review_count'] / facility_evidence['total_reviews']) * 100

    # Filter popular facilities (>= 10% of reviews AND at least 5 reviews)
    THRESHOLD_PERCENT = 10.0
    MIN_REVIEWS = 5
    popular_facilities = facility_evidence[(facility_evidence['review_percentage'] >= THRESHOLD_PERCENT) & 
                                         (facility_evidence['review_count'] >= MIN_REVIEWS)].copy()

    # Sort by place and percentage (descending)
    popular_facilities = popular_facilities.sort_values(by=['place_name', 'review_percentage'], ascending=[True, False])

    print(f"Saving popular_facilities_by_place.csv...")
    csv_columns = ['place_name', 'facility', 'review_percentage', 'review_count', 'total_reviews', 
                   'avg_confidence', 'dominant_sentiment']
    
    # Format decimals for cleaner CSV output
    popular_facilities['review_percentage'] = popular_facilities['review_percentage'].round(2)
    popular_facilities['avg_confidence'] = popular_facilities['avg_confidence'].round(4)
    
    csv_output_path = 'popular_facilities_by_place.csv'
    popular_facilities[csv_columns].to_csv(csv_output_path, index=False)

    print("Saving facility_evidence_mapping.json...")
    mapping = {}
    for _, row in popular_facilities.iterrows():
        place = row['place_name']
        facility = row['facility']
        evidence_list = row['unique_reviews']
        
        if place not in mapping:
            mapping[place] = {}
            
        mapping[place][facility] = {
            "review_percentage": row['review_percentage'],
            "review_count": row['review_count'],
            "avg_confidence": row['avg_confidence'],
            "dominant_sentiment": row['dominant_sentiment'],
            "evidence_ids": evidence_list
        }

    with open('facility_evidence_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=4)

    # Copy to data/graph/
    dest_dir = '../data/graph'
    if os.path.exists(dest_dir):
        dest_path = os.path.join(dest_dir, 'popular_facilities_by_place.csv')
        shutil.copy(csv_output_path, dest_path)
        print(f"Copied popular_facilities_by_place.csv to {dest_path}")
    else:
        print(f"Warning: Destination directory {dest_dir} does not exist.")

    print(f"Done! Extracted {len(popular_facilities)} popular facilities across {popular_facilities['place_name'].nunique()} places.")

if __name__ == "__main__":
    main()
