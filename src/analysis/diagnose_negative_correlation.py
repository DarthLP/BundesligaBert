"""
Diagnostic script to investigate negative correlation between goals_2nd/subs_2nd and announced_90.

This script analyzes potential causes:
1. Missing data bias (target_missing_90)
2. Imputation effects
3. Selection bias
4. Conditional correlations by data availability

Usage:
    python src/analysis/diagnose_negative_correlation.py
"""

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_processed_data(data_dir: Path) -> pd.DataFrame:
    """Load all processed JSON files into a DataFrame."""
    logger.info(f"Loading processed data from {data_dir}")
    
    matches = []
    season_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('season_')])
    
    for season_dir in season_dirs:
        match_files = sorted(season_dir.glob('match_*.json'))
        for match_file in match_files:
            try:
                with open(match_file, 'r', encoding='utf-8') as f:
                    match_data = json.load(f)
                
                flat_match = {}
                flat_match['match_id'] = match_data.get('match_id')
                flat_match['season'] = match_data.get('season')
                
                metadata = match_data.get('metadata', {})
                flat_match['home'] = metadata.get('home')
                flat_match['away'] = metadata.get('away')
                
                flags = match_data.get('flags', {})
                flat_match['target_missing_90'] = flags.get('target_missing_90', False)
                flat_match['is_inferred_zero_90'] = flags.get('is_inferred_zero_90', False)
                flat_match['is_imputed_announced_90'] = flags.get('is_imputed_announced_90', False)
                
                targets = match_data.get('targets', {})
                flat_match['announced_90'] = targets.get('announced_90')
                flat_match['actual_90'] = targets.get('actual_90')
                flat_match['excess_90'] = targets.get('excess_90')
                
                features_regular = match_data.get('features_regular', {})
                flat_match['goals_2nd'] = features_regular.get('goals_2nd', 0)
                flat_match['subs_2nd'] = features_regular.get('subs_2nd', 0)
                flat_match['cards_2nd'] = features_regular.get('cards_2nd', 0)
                
                matches.append(flat_match)
            except Exception as e:
                logger.warning(f"Failed to load {match_file}: {e}")
                continue
    
    df = pd.DataFrame(matches)
    logger.info(f"Loaded {len(df)} matches")
    return df


def analyze_correlation_by_missing_status(df: pd.DataFrame):
    """Analyze correlations separately for matches with/without missing announced_90."""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS BY MISSING DATA STATUS")
    print("="*80)
    
    # Overall correlation (only non-null announced_90)
    df_valid = df[df['announced_90'].notna()].copy()
    print(f"\n1. ALL MATCHES WITH VALID announced_90 (n={len(df_valid)}):")
    print(f"   Correlation goals_2nd vs announced_90: {df_valid['goals_2nd'].corr(df_valid['announced_90']):.4f}")
    print(f"   Correlation subs_2nd vs announced_90: {df_valid['subs_2nd'].corr(df_valid['announced_90']):.4f}")
    
    # Split by target_missing_90 flag
    df_not_missing = df_valid[~df_valid['target_missing_90']].copy()
    df_was_missing = df_valid[df_valid['target_missing_90']].copy()
    
    print(f"\n2. MATCHES WHERE target_missing_90 = False (n={len(df_not_missing)}):")
    if len(df_not_missing) > 0:
        print(f"   Correlation goals_2nd vs announced_90: {df_not_missing['goals_2nd'].corr(df_not_missing['announced_90']):.4f}")
        print(f"   Correlation subs_2nd vs announced_90: {df_not_missing['subs_2nd'].corr(df_not_missing['announced_90']):.4f}")
        print(f"   Mean goals_2nd: {df_not_missing['goals_2nd'].mean():.2f}")
        print(f"   Mean subs_2nd: {df_not_missing['subs_2nd'].mean():.2f}")
        print(f"   Mean announced_90: {df_not_missing['announced_90'].mean():.2f}")
    
    print(f"\n3. MATCHES WHERE target_missing_90 = True (n={len(df_was_missing)}):")
    if len(df_was_missing) > 0:
        print(f"   Correlation goals_2nd vs announced_90: {df_was_missing['goals_2nd'].corr(df_was_missing['announced_90']):.4f}")
        print(f"   Correlation subs_2nd vs announced_90: {df_was_missing['subs_2nd'].corr(df_was_missing['announced_90']):.4f}")
        print(f"   Mean goals_2nd: {df_was_missing['goals_2nd'].mean():.2f}")
        print(f"   Mean subs_2nd: {df_was_missing['subs_2nd'].mean():.2f}")
        print(f"   Mean announced_90: {df_was_missing['announced_90'].mean():.2f}")
    
    # Check if high-scoring games are more likely to have missing data
    print(f"\n4. MISSING DATA RATE BY GOALS_2ND:")
    for goals in sorted(df['goals_2nd'].unique())[:10]:  # First 10 unique values
        subset = df[df['goals_2nd'] == goals]
        missing_rate = subset['target_missing_90'].mean() * 100
        print(f"   goals_2nd = {goals}: {missing_rate:.1f}% missing (n={len(subset)})")
    
    print(f"\n5. MISSING DATA RATE BY SUBS_2ND:")
    for subs in sorted(df['subs_2nd'].unique())[:10]:
        subset = df[df['subs_2nd'] == subs]
        missing_rate = subset['target_missing_90'].mean() * 100
        print(f"   subs_2nd = {subs}: {missing_rate:.1f}% missing (n={len(subset)})")


def analyze_by_announced_value(df: pd.DataFrame):
    """Analyze relationship between goals/subs and announced_90 values."""
    print("\n" + "="*80)
    print("ANALYSIS BY ANNOUNCED_90 VALUE")
    print("="*80)
    
    df_valid = df[df['announced_90'].notna()].copy()
    
    # Group by announced_90 value
    print("\nMean goals_2nd and subs_2nd by announced_90 value:")
    grouped = df_valid.groupby('announced_90').agg({
        'goals_2nd': ['mean', 'count'],
        'subs_2nd': 'mean'
    }).round(2)
    
    for announced_val in sorted(df_valid['announced_90'].unique())[:15]:  # First 15 values
        subset = df_valid[df_valid['announced_90'] == announced_val]
        print(f"  announced_90 = {announced_val}: "
              f"mean_goals_2nd = {subset['goals_2nd'].mean():.2f}, "
              f"mean_subs_2nd = {subset['subs_2nd'].mean():.2f}, "
              f"n = {len(subset)}")


def analyze_extreme_cases(df: pd.DataFrame):
    """Analyze extreme cases: high goals/subs with low announced_90."""
    print("\n" + "="*80)
    print("EXTREME CASES: HIGH GOALS/SUBS, LOW ANNOUNCED_90")
    print("="*80)
    
    df_valid = df[df['announced_90'].notna()].copy()
    
    # High goals, low announced
    high_goals_low_announced = df_valid[
        (df_valid['goals_2nd'] >= 3) & (df_valid['announced_90'] <= 2)
    ].copy()
    
    print(f"\nMatches with goals_2nd >= 3 AND announced_90 <= 2 (n={len(high_goals_low_announced)}):")
    if len(high_goals_low_announced) > 0:
        print(f"  Mean goals_2nd: {high_goals_low_announced['goals_2nd'].mean():.2f}")
        print(f"  Mean announced_90: {high_goals_low_announced['announced_90'].mean():.2f}")
        print(f"  % with target_missing_90: {high_goals_low_announced['target_missing_90'].mean()*100:.1f}%")
        print(f"\n  Sample matches:")
        for idx, row in high_goals_low_announced.head(10).iterrows():
            print(f"    {row['season']}: {row['home']} vs {row['away']} - "
                  f"goals_2nd={row['goals_2nd']}, announced_90={row['announced_90']}, "
                  f"missing={row['target_missing_90']}")
    
    # High subs, low announced
    high_subs_low_announced = df_valid[
        (df_valid['subs_2nd'] >= 8) & (df_valid['announced_90'] <= 2)
    ].copy()
    
    print(f"\nMatches with subs_2nd >= 8 AND announced_90 <= 2 (n={len(high_subs_low_announced)}):")
    if len(high_subs_low_announced) > 0:
        print(f"  Mean subs_2nd: {high_subs_low_announced['subs_2nd'].mean():.2f}")
        print(f"  Mean announced_90: {high_subs_low_announced['announced_90'].mean():.2f}")
        print(f"  % with target_missing_90: {high_subs_low_announced['target_missing_90'].mean()*100:.1f}%")


def main():
    """Main diagnostic function."""
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data' / 'processed'
    
    df = load_processed_data(data_dir)
    
    if df.empty:
        logger.error("No data loaded!")
        return
    
    # Basic statistics
    print("="*80)
    print("BASIC STATISTICS")
    print("="*80)
    print(f"Total matches: {len(df)}")
    print(f"Matches with valid announced_90: {df['announced_90'].notna().sum()} ({df['announced_90'].notna().mean()*100:.1f}%)")
    print(f"Matches with target_missing_90: {df['target_missing_90'].sum()} ({df['target_missing_90'].mean()*100:.1f}%)")
    print(f"Mean goals_2nd: {df['goals_2nd'].mean():.2f}")
    print(f"Mean subs_2nd: {df['subs_2nd'].mean():.2f}")
    
    # Run analyses
    analyze_correlation_by_missing_status(df)
    analyze_by_announced_value(df)
    analyze_extreme_cases(df)
    
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

