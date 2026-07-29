import pandas as pd
import math

def main():
    print("Loading data...")
    try:
        # Load the summary data for full dataset to calculate correct IDF
        df = pd.read_csv('activity_summary_by_place.csv')
        
        # Load triples to get confidence (since it's not in the summary)
        triples_df = pd.read_csv('review_activity_triples.csv')
        
        # Load popular activities to enrich them
        popular_df = pd.read_csv('popular_activities_by_place.csv')
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure 'activity_summary_by_place.csv', 'review_activity_triples.csv', and 'popular_activities_by_place.csv' exist.")
        return

    print("Calculating average confidence for full dataset...")
    activity_triples = triples_df[(triples_df['subject_type'] == 'Place') & (triples_df['object_type'] == 'Activity')].copy()
    confidence_df = activity_triples.groupby(['subject', 'object'])['confidence'].mean().reset_index()
    confidence_df.columns = ['place_name', 'activity', 'average_confidence_calc']

    # Merge confidence into full summary
    df = pd.merge(df, confidence_df, on=['place_name', 'activity'], how='left')

    print("Calculating TF-IDF on full dataset...")
    # TF: total_mentions of activity in place / total_mentions of all activities in place
    place_totals = df.groupby('place_name')['total_mentions'].sum().reset_index(name='place_total_mentions')
    df = pd.merge(df, place_totals, on='place_name')
    df['TF'] = df['total_mentions'] / df['place_total_mentions']

    # IDF: log(total number of places / number of places that have activity)
    total_places = df['place_name'].nunique()
    activity_df_counts = df.groupby('activity')['place_name'].nunique().reset_index(name='doc_freq')
    df = pd.merge(df, activity_df_counts, on='activity')

    # Use math.log for natural logarithm
    df['IDF'] = df['doc_freq'].apply(lambda x: math.log(total_places / x) if x > 0 else 0)

    # TF-IDF
    df['TF_IDF'] = df['TF'] * df['IDF']
    
    # Defining score
    df['defining_score'] = df['TF_IDF'] * df['average_confidence_calc']

    print("Enriching popular activities with metrics...")
    # Merge metrics into popular activities
    metrics_to_add = df[['place_name', 'activity', 'TF', 'IDF', 'TF_IDF', 'defining_score']]
    enriched_popular = pd.merge(popular_df, metrics_to_add, on=['place_name', 'activity'], how='left')

    # Sort by place and defining score descending
    enriched_popular = enriched_popular.sort_values(by=['place_name', 'defining_score'], ascending=[True, False])

    # Save the result
    output_file = 'popular_activities_with_use_data.csv'
    enriched_popular.to_csv(output_file, index=False)
    print(f"Saved enriched popular activities to '{output_file}'")

    print("\nSample of enriched activities:")
    print(enriched_popular[['place_name', 'activity', 'TF', 'IDF', 'TF_IDF', 'defining_score']].head())

if __name__ == "__main__":
    main()
