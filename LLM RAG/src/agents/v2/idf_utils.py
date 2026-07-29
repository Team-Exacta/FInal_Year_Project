import os
import pandas as pd
import logging
from functools import lru_cache
from src.core.config import settings

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

FEATURE_CSV = os.path.join(BASE_DIR, "data", "extraction_outputs", "feature_extraction_outputs", "highly_accurate_features.csv")
ACTIVITY_CSV = os.path.join(BASE_DIR, "data", "extraction_outputs", "activity_extraction_outputs", "activity_summary_by_place.csv")
FACILITY_CSV = os.path.join(BASE_DIR, "data", "extraction_outputs", "facility_extraction_outputs", "facility_summary_by_place.csv")

DEFAULT_THRESHOLD = 20.0

@lru_cache(maxsize=1)
def _load_frequencies():
    """
    Loads and caches the feature, activity, and facility frequencies from the CSV outputs.
    Returns a dictionary mapping term (lowercase) to its percentage frequency (0-100).
    """
    frequencies = {}
    
    def process_df(filepath, term_col):
        if not os.path.exists(filepath):
            logger.warning(f"[IDF_Utils] File not found: {filepath}")
            return
            
        try:
            df = pd.read_csv(filepath)
            if 'place_name' not in df.columns or term_col not in df.columns:
                return
                
            total_places = df['place_name'].nunique()
            if total_places == 0:
                return
                
            counts = df.groupby(term_col)['place_name'].nunique()
            for term, count in counts.items():
                freq = (count / total_places) * 100
                frequencies[str(term).lower()] = freq
                
        except Exception as e:
            logger.error(f"[IDF_Utils] Error processing {filepath}: {e}")

    process_df(FEATURE_CSV, 'feature')
    process_df(ACTIVITY_CSV, 'activity')
    process_df(FACILITY_CSV, 'facility')
    
    return frequencies

def get_dynamic_threshold(terms, base_threshold=20.0, min_threshold=0.0):
    """
    Calculates the dynamic percentage threshold required for a list of terms.
    If multiple terms are provided, it bases the threshold on the RAREST term.
    
    If settings.use_dynamic_thresholds is False, it returns base_threshold.
    """
    if not settings.use_dynamic_thresholds:
        return base_threshold
        
    if not terms:
        return base_threshold
        
    frequencies = _load_frequencies()
    
    min_freq = 100.0
    for term in terms:
        term_lower = term.lower()
        if term_lower in frequencies:
            min_freq = min(min_freq, frequencies[term_lower])
        else:
            # If a term is literally not in the dataset at all, it's extremely rare (freq = 0)
            min_freq = 0.0
            
    # Calculate threshold based on rarity
    if min_freq >= 30.0:
        return base_threshold
    elif min_freq <= 5.0:
        return min_threshold
    else:
        # Quadratic interpolation between 5% and 30% (Robertson BM25 alignment)
        # e.g., if min_freq is 15%, val = (((15-5)/25)**2) * 20 = 3.2%
        val = (((min_freq - 5.0) / 25.0) ** 2) * base_threshold
        return round(val, 2)
