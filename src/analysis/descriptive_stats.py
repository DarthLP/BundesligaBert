"""
Descriptive Statistics and Visualization Script for BundesligaBERT

This script loads preprocessed match data and generates comprehensive descriptive
statistics and visualizations for the term paper. It analyzes data integrity,
target distributions, causal drivers, text quality, and behavioral patterns.

The script focuses on:
- Data quality checks (imputation rates, missing data, ghost games)
- Target variable analysis (announced vs actual time)
- Excess time analysis (actual - announced)
- Event distributions (goals, subs, cards, VAR, injuries, disturbances)
- Overtime analysis
- Home bias and crowd pressure effects

Usage:
    # Run from project root
    python src/analysis/descriptive_stats.py
    
    # Or as module
    python -m src.analysis.descriptive_stats

Input:
    Processed JSON files from data/processed/season_{season}/match_{match_id}.json

Output:
    - Figures: reports/processed_data/figures/*.png
    - Tables: reports/processed_data/tables/*.csv
    - Report: reports/processed_data/tables/descriptive_report.txt

Author: BundesligaBERT Project
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seaborn style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def load_processed_data(data_dir: Path) -> pd.DataFrame:
    """
    Load all processed JSON files and flatten into a DataFrame.
    
    Recursively loads all processed JSON files from data/processed/season_*/
    and flattens nested dictionaries into flat columns.
    
    Args:
        data_dir: Path to data/processed directory
        
    Returns:
        DataFrame with one row per match, all features flattened
    """
    logger.info(f"Loading processed data from {data_dir}")
    
    matches = []
    processed_count = 0
    
    # Find all season directories
    season_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('season_')])
    
    for season_dir in season_dirs:
        logger.info(f"Processing {season_dir.name}")
        match_files = sorted(season_dir.glob('match_*.json'))
        
        for match_file in match_files:
            try:
                with open(match_file, 'r', encoding='utf-8') as f:
                    match_data = json.load(f)
                
                # Flatten the nested structure
                flat_match = {}
                
                # Basic fields
                flat_match['match_id'] = match_data.get('match_id')
                flat_match['season'] = match_data.get('season')
                
                # Flatten metadata
                metadata = match_data.get('metadata', {})
                flat_match['home'] = metadata.get('home')
                flat_match['away'] = metadata.get('away')
                flat_match['attendance'] = metadata.get('attendance')
                flat_match['is_ghost_game'] = metadata.get('is_ghost_game', False)  # Can be True, False, or None
                flat_match['matchday'] = metadata.get('matchday')
                flat_match['final_score'] = metadata.get('final_score')
                
                # Flatten flags (new structure)
                flags = match_data.get('flags', {})
                flat_match['is_inferred_zero_45'] = flags.get('is_inferred_zero_45', False)
                flat_match['is_inferred_zero_90'] = flags.get('is_inferred_zero_90', False)
                flat_match['target_missing_45'] = flags.get('target_missing_45', False)
                flat_match['target_missing_90'] = flags.get('target_missing_90', False)
                flat_match['is_imputed_actual_45'] = flags.get('is_imputed_actual_45', False)
                flat_match['is_imputed_actual_90'] = flags.get('is_imputed_actual_90', False)
                flat_match['is_imputed_announced_45'] = flags.get('is_imputed_announced_45', False)
                flat_match['is_imputed_announced_90'] = flags.get('is_imputed_announced_90', False)
                flat_match['missing_cutoff_markers'] = flags.get('missing_cutoff_markers')
                flat_match['score_timeline_valid'] = flags.get('score_timeline_valid', True)
                
                # Flatten targets (new structure with excess and raw)
                targets = match_data.get('targets', {})
                flat_match['announced_45'] = targets.get('announced_45')
                flat_match['actual_45'] = targets.get('actual_45')
                flat_match['excess_45'] = targets.get('excess_45')
                flat_match['announced_90'] = targets.get('announced_90')
                flat_match['actual_90'] = targets.get('actual_90')
                flat_match['excess_90'] = targets.get('excess_90')
                
                # Flatten raw targets
                targets_raw = targets.get('targets_raw', {})
                flat_match['raw_announced_time_45'] = targets_raw.get('announced_time_45')
                flat_match['raw_announced_time_90'] = targets_raw.get('announced_time_90')
                flat_match['raw_actual_played_45'] = targets_raw.get('actual_played_45')
                flat_match['raw_actual_played_90'] = targets_raw.get('actual_played_90')
                
                # Flatten features_regular
                features_regular = match_data.get('features_regular', {})
                for key, value in features_regular.items():
                    flat_match[key] = value
                
                # Flatten features_overtime
                features_overtime = match_data.get('features_overtime', {})
                for key, value in features_overtime.items():
                    flat_match[key] = value
                
                matches.append(flat_match)
                processed_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to load {match_file}: {e}")
                continue
    
    logger.info(f"Loaded {processed_count} matches")
    
    df = pd.DataFrame(matches)
    
    # Use excess_45 and excess_90 if available, otherwise calculate them
    if 'excess_45' in df.columns:
        df['excess_time_45'] = df['excess_45']
    else:
        # Fallback: calculate if not present
        df['excess_time_45'] = df['actual_45'] - df['announced_45']
    
    if 'excess_90' in df.columns:
        df['excess_time_90'] = df['excess_90']
    else:
        # Fallback: calculate if not present
        df['excess_time_90'] = df['actual_90'] - df['announced_90']
    
    # Calculate score state at minute 90 (exclude None/null values)
    # Only calculate where both home_score_90 and away_score_90 are not None/null
    df['score_state_90'] = None
    valid_scores = df[df['home_score_90'].notna() & df['away_score_90'].notna()]
    df.loc[valid_scores.index, 'score_state_90'] = valid_scores.apply(
        lambda row: 'Home Leading' if row['home_score_90'] > row['away_score_90']
        else ('Away Leading' if row['away_score_90'] > row['home_score_90'] else 'Draw'),
        axis=1
    )
    
    # Calculate overtime events total (including injuries and disturbances)
    df['ot_events_total'] = (
        df['ot_goals_90'].fillna(0) + 
        df['ot_subs_90'].fillna(0) + 
        df['ot_var_90'].fillna(0) +
        df['ot_injuries_90'].fillna(0) +
        df['ot_disturbances_90'].fillna(0)
    )
    
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def plot_missing_data_analysis(df: pd.DataFrame, output_dir: Path):
    """
    Generate CSV with missing data statistics and list of affected files.
    
    Args:
        df: DataFrame with match data
        output_dir: Directory to save CSV
    """
    logger.info("Generating missing data analysis CSV")
    
    # Count missing data per season
    missing_stats = df.groupby('season').agg({
        'missing_cutoff_markers': lambda x: (x.notna()).sum(),
        'score_timeline_valid': lambda x: (~x).sum()
    }).reset_index()
    
    missing_stats.columns = ['season', 'missing_cutoff_markers', 'invalid_score_timeline']
    
    # Get list of affected match_ids with file paths
    missing_cutoff_matches = df[df['missing_cutoff_markers'].notna()][['match_id', 'season', 'missing_cutoff_markers']].copy()
    missing_cutoff_matches['file_path'] = missing_cutoff_matches.apply(
        lambda row: f"data/processed/season_{row['season']}/match_{row['match_id']}.json", axis=1
    )
    
    invalid_timeline_matches = df[~df['score_timeline_valid']][['match_id', 'season']].copy()
    invalid_timeline_matches['issue'] = 'invalid_score_timeline'
    invalid_timeline_matches['file_path'] = invalid_timeline_matches.apply(
        lambda row: f"data/processed/season_{row['season']}/match_{row['match_id']}.json", axis=1
    )
    
    # Save summary stats
    table_path = output_dir.parent / 'tables' / 'missing_data_stats.csv'
    table_path.parent.mkdir(parents=True, exist_ok=True)
    missing_stats.to_csv(table_path, index=False)
    
    # Save list of affected files
    if len(missing_cutoff_matches) > 0:
        cutoff_path = output_dir.parent / 'tables' / 'missing_cutoff_markers_matches.csv'
        missing_cutoff_matches.to_csv(cutoff_path, index=False)
        logger.info(f"Saved {len(missing_cutoff_matches)} matches with missing cutoff markers")
    
    if len(invalid_timeline_matches) > 0:
        timeline_path = output_dir.parent / 'tables' / 'invalid_score_timeline_matches.csv'
        invalid_timeline_matches.to_csv(timeline_path, index=False)
        logger.info(f"Saved {len(invalid_timeline_matches)} matches with invalid score timeline")
    
    logger.info("Saved missing_data_stats.csv")


def plot_matches_per_season(df: pd.DataFrame, output_dir: Path):
    """
    Plot match counts per season.
    
    Args:
        df: DataFrame with match data
        output_dir: Directory to save plots
    """
    logger.info("Plotting matches per season")
    
    match_counts = df.groupby('season').size().reset_index(name='count')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(match_counts['season'], match_counts['count'], color='steelblue')
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Number of Matches', fontsize=12)
    ax.set_title('Matches per Season', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    # Add expected line (306 matches)
    ax.axhline(y=306, color='r', linestyle='--', label='Expected (306)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'matches_per_season.png', bbox_inches='tight')
    plt.close()
    
    logger.info("Saved matches_per_season.png")


def plot_ghost_games_analysis(df: pd.DataFrame, output_dir: Path):
    """
    Plot ghost games analysis (COVID-19 impact).
    
    Args:
        df: DataFrame with match data
        output_dir: Directory to save plots
    """
    logger.info("Plotting ghost games analysis")
    
    # Handle None values - convert to numeric for aggregation (True=1, False=0, None=1 to count as ghost game)
    df_ghost_numeric = df.copy()
    df_ghost_numeric['is_ghost_game_numeric'] = df_ghost_numeric['is_ghost_game'].map({True: 1, False: 0, None: 1}).astype(float)
    
    ghost_stats = df_ghost_numeric.groupby('season').agg({
        'is_ghost_game_numeric': ['sum', 'count', 'mean']
    }).reset_index()
    
    ghost_stats.columns = ['season', 'ghost_count', 'total_matches', 'ghost_pct']
    ghost_stats['ghost_pct'] = ghost_stats['ghost_pct'] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart: count
    ax1.bar(ghost_stats['season'], ghost_stats['ghost_count'], color='darkred')
    ax1.set_xlabel('Season', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Ghost Games Count per Season', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Line chart: percentage
    ax2.plot(ghost_stats['season'], ghost_stats['ghost_pct'], marker='o', linewidth=2, markersize=8, color='darkred')
    ax2.set_xlabel('Season', fontsize=12)
    ax2.set_ylabel('% Ghost Games', fontsize=12)
    ax2.set_title('Ghost Games Percentage per Season', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ghost_games_analysis.png', bbox_inches='tight')
    plt.close()
    
    # Save table
    table_path = output_dir.parent / 'tables' / 'ghost_games_stats.csv'
    table_path.parent.mkdir(parents=True, exist_ok=True)
    ghost_stats.to_csv(table_path, index=False)
    
    logger.info("Saved ghost_games_analysis.png and ghost_games_stats.csv")


def plot_announced_time_distribution(df: pd.DataFrame, output_dir: Path):
    """
    Plot distributions of announced and actual time for 45 and 90.
    
    Args:
        df: DataFrame with match data
        output_dir: Directory to save plots
    """
    logger.info("Plotting announced and actual time distributions")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Filter out null values
    announced_45 = df['announced_45'].dropna()
    announced_90 = df['announced_90'].dropna()
    actual_45 = df['actual_45'].dropna()
    actual_90 = df['actual_90'].dropna()
    
    # Plot Announced 45
    ax1 = axes[0, 0]
    counts, bins, patches = ax1.hist(announced_45, bins=range(0, int(announced_45.max()) + 2), 
                                     color='steelblue', edgecolor='black')
    # Add counts above bars
    for count, patch in zip(counts, patches):
        if count > 0:
            ax1.text(patch.get_x() + patch.get_width()/2., count,
                    f'{int(count)}', ha='center', va='bottom', fontsize=8)
    ax1.set_xlabel('Announced Time (minutes)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Announced Time: First Half (45)', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot Announced 90
    ax2 = axes[0, 1]
    counts, bins, patches = ax2.hist(announced_90, bins=range(0, int(announced_90.max()) + 2), 
                                     color='coral', edgecolor='black')
    for count, patch in zip(counts, patches):
        if count > 0:
            ax2.text(patch.get_x() + patch.get_width()/2., count,
                    f'{int(count)}', ha='center', va='bottom', fontsize=8)
    ax2.set_xlabel('Announced Time (minutes)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Announced Time: Second Half (90)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot Actual 45
    ax3 = axes[1, 0]
    if len(actual_45) > 0:
        counts, bins, patches = ax3.hist(actual_45, bins=range(0, int(actual_45.max()) + 2), 
                                         color='green', edgecolor='black')
        for count, patch in zip(counts, patches):
            if count > 0:
                ax3.text(patch.get_x() + patch.get_width()/2., count,
                        f'{int(count)}', ha='center', va='bottom', fontsize=8)
    ax3.set_xlabel('Actual Time (minutes)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distribution of Actual Time: First Half (45)', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot Actual 90
    ax4 = axes[1, 1]
    if len(actual_90) > 0:
        counts, bins, patches = ax4.hist(actual_90, bins=range(0, int(actual_90.max()) + 2), 
                                        color='purple', edgecolor='black')
        for count, patch in zip(counts, patches):
            if count > 0:
                ax4.text(patch.get_x() + patch.get_width()/2., count,
                        f'{int(count)}', ha='center', va='bottom', fontsize=8)
    ax4.set_xlabel('Actual Time (minutes)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Distribution of Actual Time: Second Half (90)', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'announced_time_distribution.png', bbox_inches='tight')
    plt.close()
    
    logger.info("Saved announced_time_distribution.png")


def plot_time_creep(df: pd.DataFrame, output_dir: Path):
    """
    Plot time creep: mean announced vs actual time over seasons.
    
    Args:
        df: DataFrame with match data
        output_dir: Directory to save plots
    """
    logger.info("Plotting time creep")
    
    time_stats = df.groupby('season').agg({
        'announced_90': 'mean',
        'actual_90': 'mean'
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(time_stats['season'], time_stats['announced_90'], marker='o', linewidth=2, markersize=8, label='Announced', color='steelblue')
    ax.plot(time_stats['season'], time_stats['actual_90'], marker='s', linewidth=2, markersize=8, label='Actual', color='coral')
    
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Mean Time (minutes)', fontsize=12)
    ax.set_title('Time Creep: Mean Announced vs Actual Time (90) per Season', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'time_creep.png', bbox_inches='tight')
    plt.close()
    
    logger.info("Saved time_creep.png")


def plot_excess_time_distribution(df: pd.DataFrame, output_dir: Path):
    """
    Plot distribution of excess_time_45 and excess_time_90.
    
    Args:
        df: DataFrame with match data
        output_dir: Directory to save plots
    """
    logger.info("Plotting excess time distribution")
    
    excess_45 = df['excess_time_45'].dropna()
    excess_90 = df['excess_time_90'].dropna()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot for 45
    if len(excess_45) > 0:
        ax1.hist(excess_45, bins=range(int(excess_45.min()), int(excess_45.max()) + 2), 
                color='steelblue', edgecolor='black')
        ax1.set_xlabel('Excess Time (minutes)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of Excess Time: First Half (45)', fontsize=14, fontweight='bold')
        ax1.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero excess')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
    
    # Plot for 90
    if len(excess_90) > 0:
        ax2.hist(excess_90, bins=range(int(excess_90.min()), int(excess_90.max()) + 2), 
                color='coral', edgecolor='black')
        ax2.set_xlabel('Excess Time (minutes)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Excess Time: Second Half (90)', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero excess')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'excess_time_distribution.png', bbox_inches='tight')
    plt.close()
    
    logger.info("Saved excess_time_distribution.png")


def table_excess_time_stats(df: pd.DataFrame, output_dir: Path):
    """
    Generate table of excess time statistics per season for both 45 and 90.
    
    Args:
        df: DataFrame with match data
        output_dir: Directory to save tables
    """
    logger.info("Generating excess time stats table")
    
    # Stats for 45
    excess_stats_45 = df.groupby('season')['excess_time_45'].agg([
        'min', 'max', 'mean', 'std'
    ]).reset_index()
    excess_stats_45.columns = ['season', 'min_45', 'max_45', 'mean_45', 'std_45']
    excess_stats_45 = excess_stats_45.round(2)
    
    # Stats for 90
    excess_stats_90 = df.groupby('season')['excess_time_90'].agg([
        'min', 'max', 'mean', 'std'
    ]).reset_index()
    excess_stats_90.columns = ['season', 'min_90', 'max_90', 'mean_90', 'std_90']
    excess_stats_90 = excess_stats_90.round(2)
    
    # Merge
    excess_stats = excess_stats_45.merge(excess_stats_90, on='season')
    
    table_path = output_dir.parent / 'tables' / 'excess_time_stats.csv'
    table_path.parent.mkdir(parents=True, exist_ok=True)
    excess_stats.to_csv(table_path, index=False)
    
    logger.info(f"Saved excess_time_stats.csv")


def plot_event_distributions_combined(df: pd.DataFrame, output_dir: Path):
    """
    Plot combined event distributions for goals, subs, cards, VAR, injuries, disturbances.
    
    Args:
        df: DataFrame with match data
        output_dir: Directory to save plots
    """
    logger.info("Plotting combined event distributions")
    
    event_types = [
        ('goals', ['goals_1st', 'goals_2nd', 'ot_goals_45', 'ot_goals_90'], 'Goals'),
        ('subs', ['subs_1st', 'subs_2nd', 'ot_subs_45', 'ot_subs_90'], 'Substitutions'),
        ('cards', ['cards_1st', 'cards_2nd', 'ot_cards_45', 'ot_cards_90'], 'Cards'),
        ('var', ['var_1st', 'var_2nd', 'ot_var_45', 'ot_var_90'], 'VAR'),
        ('injuries', ['injuries_1st', 'injuries_2nd', 'ot_injuries_45', 'ot_injuries_90'], 'Injuries'),
        ('disturbances', ['disturbances_1st', 'disturbances_2nd', 'ot_disturbances_45', 'ot_disturbances_90'], 'Disturbances')
    ]
    
    for event_key, columns, event_name in event_types:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data - exclude None/null values (use only valid data)
        data_to_plot = []
        labels = []
        for col in columns:
            if col in df.columns:
                # Only use non-null values (exclude None/null, but keep 0s)
                valid_data = df[col].dropna().values
                if len(valid_data) > 0:
                    data_to_plot.append(valid_data)
                    labels.append(col.replace('_', ' ').title())
        
        if len(data_to_plot) == 0:
            logger.warning(f"No valid data for {event_name}, skipping plot")
            continue
        
        # Create grouped bar chart
        x = np.arange(len(labels))
        width = 0.6
        
        # Calculate means and standard errors (for error bars)
        means = [np.mean(d) for d in data_to_plot]
        stds = [np.std(d) / np.sqrt(len(d)) for d in data_to_plot]  # Standard error
        
        # Ensure error bars don't go below 0
        # yerr expects [lower_distance, upper_distance] from the mean
        lower_errors = [min(mean, std) for mean, std in zip(means, stds)]  # Distance down (capped at mean)
        upper_errors = stds  # Distance up
        
        bars = ax.bar(x, means, width, yerr=[lower_errors, upper_errors], capsize=5, 
                     color='steelblue', edgecolor='black', alpha=0.7, error_kw={'elinewidth': 1.5})
        ax.set_xlabel('Match Phase', fontsize=12)
        ax.set_ylabel('Mean Count', fontsize=12)
        ax.set_title(f'Mean {event_name} by Match Phase', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(bottom=0)  # Start y-axis at 0
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'event_distribution_{event_key}.png', bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved event_distribution_{event_key}.png")


# Removed: plot_var_impact, plot_injury_impact, plot_card_impact
# These plots were not providing clear insights and were removed per user request.


def plot_overtime_chaos(df: pd.DataFrame, output_dir: Path):
    """
    Plot overtime chaos: average OT events per OT minute per season (including VAR, injuries and disturbances).
    
    Only includes games with actual_90 > 90 (positive overtime).
    If actual_90 is None, the game is excluded.
    If we don't have a value for ot_*_45 but we do for ot_*_90, still use the 90 minute data.
    
    Calculates: (sum of OT events) / (sum of OT minutes) for each event type per season.
    OT minutes = max(0, actual_90 - 90) where actual_90 is not None.
    
    Args:
        df: DataFrame with match data
        output_dir: Directory to save plots
    """
    logger.info("Plotting overtime chaos (events per OT minute)")
    
    # Work with a copy to avoid modifying the original
    df_ot = df.copy()
    
    # Calculate OT minutes per game (only where actual_90 is not None)
    # OT minutes = max(0, actual_90 - 90)
    df_ot['ot_minutes'] = df_ot.apply(
        lambda row: max(0, row['actual_90'] - 90) if pd.notna(row['actual_90']) else None,
        axis=1
    )
    
    # Filter: Only include games with positive OT minutes (actual_90 > 90)
    # This excludes games where actual_90 is None or actual_90 <= 90
    df_ot_positive = df_ot[df_ot['ot_minutes'] > 0].copy()
    
    if len(df_ot_positive) == 0:
        logger.warning("No games with positive overtime found, skipping overtime chaos plot")
        return
    
    # For each OT column, use 90 if available, otherwise use 45 if available
    ot_cols_90 = ['ot_goals_90', 'ot_subs_90', 'ot_cards_90', 'ot_var_90', 'ot_injuries_90', 'ot_disturbances_90']
    ot_cols_45 = ['ot_goals_45', 'ot_subs_45', 'ot_cards_45', 'ot_var_45', 'ot_injuries_45', 'ot_disturbances_45']
    
    # For each event type, prefer 90, fallback to 45 if 90 is missing
    for i, col_90 in enumerate(ot_cols_90):
        if col_90 in df_ot_positive.columns and ot_cols_45[i] in df_ot_positive.columns:
            # Where 90 is missing but 45 is available, use 45
            mask = df_ot_positive[col_90].isna() & df_ot_positive[ot_cols_45[i]].notna()
            df_ot_positive.loc[mask, col_90] = df_ot_positive.loc[mask, ot_cols_45[i]]
    
    # Fill missing OT event values with 0 (for games with OT, missing means 0 events)
    for col in ot_cols_90:
        if col in df_ot_positive.columns:
            df_ot_positive[col] = df_ot_positive[col].fillna(0)
    
    # Calculate events per OT minute per season
    # For each season: sum(events) / sum(OT_minutes)
    ot_stats = df_ot_positive.groupby('season').agg({
        'ot_goals_90': 'sum',
        'ot_subs_90': 'sum',
        'ot_cards_90': 'sum',
        'ot_var_90': 'sum',
        'ot_injuries_90': 'sum',
        'ot_disturbances_90': 'sum',
        'ot_minutes': 'sum'
    }).reset_index()
    
    # Calculate rates (events per OT minute)
    ot_stats['goals_per_ot_min'] = ot_stats['ot_goals_90'] / ot_stats['ot_minutes']
    ot_stats['subs_per_ot_min'] = ot_stats['ot_subs_90'] / ot_stats['ot_minutes']
    ot_stats['cards_per_ot_min'] = ot_stats['ot_cards_90'] / ot_stats['ot_minutes']
    ot_stats['var_per_ot_min'] = ot_stats['ot_var_90'] / ot_stats['ot_minutes']
    ot_stats['injuries_per_ot_min'] = ot_stats['ot_injuries_90'] / ot_stats['ot_minutes']
    ot_stats['disturbances_per_ot_min'] = ot_stats['ot_disturbances_90'] / ot_stats['ot_minutes']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(ot_stats))
    width = 0.6
    
    # Stacked bars showing events per OT minute
    ax.bar(x, ot_stats['goals_per_ot_min'], width, label='Goals', color='green', alpha=0.7)
    ax.bar(x, ot_stats['subs_per_ot_min'], width, bottom=ot_stats['goals_per_ot_min'], label='Subs', color='blue', alpha=0.7)
    ax.bar(x, ot_stats['cards_per_ot_min'], width, 
           bottom=ot_stats['goals_per_ot_min'] + ot_stats['subs_per_ot_min'], 
           label='Cards', color='red', alpha=0.7)
    ax.bar(x, ot_stats['var_per_ot_min'], width,
           bottom=ot_stats['goals_per_ot_min'] + ot_stats['subs_per_ot_min'] + ot_stats['cards_per_ot_min'],
           label='VAR', color='yellow', alpha=0.7)
    ax.bar(x, ot_stats['injuries_per_ot_min'], width,
           bottom=ot_stats['goals_per_ot_min'] + ot_stats['subs_per_ot_min'] + ot_stats['cards_per_ot_min'] + ot_stats['var_per_ot_min'],
           label='Injuries', color='orange', alpha=0.7)
    ax.bar(x, ot_stats['disturbances_per_ot_min'], width,
           bottom=ot_stats['goals_per_ot_min'] + ot_stats['subs_per_ot_min'] + ot_stats['cards_per_ot_min'] + ot_stats['var_per_ot_min'] + ot_stats['injuries_per_ot_min'],
           label='Disturbances', color='purple', alpha=0.7)
    
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Average Events per OT Minute', fontsize=12)
    ax.set_title('Overtime Chaos: Average OT Events per OT Minute per Season\n(Only games with actual_90 > 90)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ot_stats['season'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overtime_chaos.png', bbox_inches='tight')
    plt.close()
    
    logger.info("Saved overtime_chaos.png")


def plot_excess_vs_ot_events(df: pd.DataFrame, output_dir: Path):
    """
    Plot excess time vs overtime events (including injuries and disturbances).
    
    Args:
        df: DataFrame with match data
        output_dir: Directory to save plots
    """
    logger.info("Plotting excess vs OT events")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(df['ot_events_total'], df['excess_time_90'], alpha=0.5, color='steelblue')
    
    # Add trend line
    z = np.polyfit(df['ot_events_total'].fillna(0), df['excess_time_90'].fillna(0), 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['ot_events_total'].min(), df['ot_events_total'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, 
           label=f'Linear fit: y = {z[0]:.2f}x + {z[1]:.2f}')
    
    ax.set_xlabel('Overtime Events Total (goals + subs + VAR + injuries + disturbances)', fontsize=12)
    ax.set_ylabel('Excess Time (minutes)', fontsize=12)
    ax.set_title('Excess Time vs Overtime Events', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'excess_vs_ot_events.png', bbox_inches='tight')
    plt.close()
    
    logger.info("Saved excess_vs_ot_events.png")


def plot_home_bias_raw(df: pd.DataFrame, output_dir: Path):
    """
    Plot home bias: excess time by score state.
    Creates 2 combined graphs:
    - 1 for processed targets: all matches, ghost games only, non-ghost games only
    - 1 for raw targets: all matches, ghost games only, non-ghost games only
    
    Includes mean and standard deviation for each group.
    Raw excess time calculation: raw_actual_played_90 - raw_announced_time_90
    None values are excluded (only matches with both raw values available are included).
    Processed excess time: actual_90 - announced_90 (None values handled by imputation logic).
    
    Args:
        df: DataFrame with match data
        output_dir: Directory to save plots
    """
    logger.info("Plotting home bias (raw and processed, combined)")
    
    # Calculate raw excess time where available (exclude None values)
    # Only calculate where both raw_actual_played_90 and raw_announced_time_90 are not None
    df['raw_excess_time_90'] = df.apply(
        lambda row: row['raw_actual_played_90'] - row['raw_announced_time_90'] 
        if pd.notna(row['raw_actual_played_90']) and pd.notna(row['raw_announced_time_90']) 
        else None, 
        axis=1
    )
    
    # Define order and colors
    order = ['Home Leading', 'Draw', 'Away Leading']
    colors = ['green', 'gray', 'red']
    
    # ===== PROCESSED TARGETS =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Processed targets - ALL matches
    ax = axes[0]
    bias_stats = df.groupby('score_state_90')['excess_time_90'].agg(['mean', 'std']).reset_index()
    bias_stats = bias_stats.set_index('score_state_90').reindex(order).reset_index()
    x_pos = np.arange(len(bias_stats))
    
    bars = ax.bar(x_pos, bias_stats['mean'], 
                  color=colors, edgecolor='black', alpha=0.7)
    max_height = bias_stats['mean'].max()
    min_height = bias_stats['mean'].min()
    y_range = max_height - min_height if max_height != min_height else 1
    y_margin = max(0.3, y_range * 0.15)  # 15% margin or at least 0.3
    
    for i, (bar, mean_val) in enumerate(zip(bars, bias_stats['mean'])):
        height = bar.get_height()
        text_y = height + y_margin * 0.3 if height >= 0 else height - y_margin * 0.3
        ax.text(bar.get_x() + bar.get_width()/2., text_y,
               f'{mean_val:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bias_stats['score_state_90'], fontsize=11)
    ax.set_ylabel('Mean Excess Time (minutes)', fontsize=12)
    ax.set_title('Processed Targets: ALL Matches\n(n={})'.format(len(df)), 
                fontsize=13, fontweight='bold')
    ax.set_ylim(min_height - y_margin, max_height + y_margin)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 2. Processed targets - Ghost games ONLY
    ax = axes[1]
    df_ghost = df[df['is_ghost_game'] != False]  # Includes True and None (missing attendance data)
    if len(df_ghost) > 0:
        bias_stats = df_ghost.groupby('score_state_90')['excess_time_90'].agg(['mean', 'std']).reset_index()
        bias_stats = bias_stats.set_index('score_state_90').reindex(order).reset_index()
        x_pos = np.arange(len(bias_stats))
        
        bars = ax.bar(x_pos, bias_stats['mean'], 
                     color=colors, edgecolor='black', alpha=0.7)
        max_height = bias_stats['mean'].max()
        min_height = bias_stats['mean'].min()
        y_range = max_height - min_height if max_height != min_height else 1
        y_margin = max(0.3, y_range * 0.15)  # 15% margin or at least 0.3
        
        for i, (bar, mean_val) in enumerate(zip(bars, bias_stats['mean'])):
            height = bar.get_height()
            text_y = height + y_margin * 0.3 if height >= 0 else height - y_margin * 0.3
            ax.text(bar.get_x() + bar.get_width()/2., text_y,
                   f'{mean_val:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bias_stats['score_state_90'], fontsize=11)
        ax.set_ylim(min_height - y_margin, max_height + y_margin)
    else:
        ax.set_ylim(-1, 1)  # Default range if no data
    ax.set_ylabel('Mean Excess Time (minutes)', fontsize=12)
    ax.set_title('Processed Targets: Ghost Games ONLY\n(n={})'.format(len(df_ghost)), 
                fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 3. Processed targets - Non-ghost games ONLY
    ax = axes[2]
    df_no_ghost = df[df['is_ghost_game'] == False]  # Only False, exclude True and None
    if len(df_no_ghost) > 0:
        bias_stats = df_no_ghost.groupby('score_state_90')['excess_time_90'].agg(['mean', 'std']).reset_index()
        bias_stats = bias_stats.set_index('score_state_90').reindex(order).reset_index()
        x_pos = np.arange(len(bias_stats))
        
        bars = ax.bar(x_pos, bias_stats['mean'], 
                     color=colors, edgecolor='black', alpha=0.7)
        max_height = bias_stats['mean'].max()
        min_height = bias_stats['mean'].min()
        y_range = max_height - min_height if max_height != min_height else 1
        y_margin = max(0.3, y_range * 0.15)  # 15% margin or at least 0.3
        
        for i, (bar, mean_val) in enumerate(zip(bars, bias_stats['mean'])):
            height = bar.get_height()
            text_y = height + y_margin * 0.3 if height >= 0 else height - y_margin * 0.3
            ax.text(bar.get_x() + bar.get_width()/2., text_y,
                   f'{mean_val:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bias_stats['score_state_90'], fontsize=11)
        ax.set_ylim(min_height - y_margin, max_height + y_margin)
    else:
        ax.set_ylim(-1, 1)  # Default range if no data
    ax.set_ylabel('Mean Excess Time (minutes)', fontsize=12)
    ax.set_title('Processed Targets: Non-Ghost Games ONLY\n(n={})'.format(len(df_no_ghost)), 
                fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'home_bias_processed.png', bbox_inches='tight')
    plt.close()
    
    # ===== RAW TARGETS =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 4. Raw targets - ALL matches
    ax = axes[0]
    df_raw = df[df['raw_excess_time_90'].notna()]
    if len(df_raw) > 0:
        bias_stats = df_raw.groupby('score_state_90')['raw_excess_time_90'].agg(['mean', 'std']).reset_index()
        bias_stats = bias_stats.set_index('score_state_90').reindex(order).reset_index()
        x_pos = np.arange(len(bias_stats))
        
        bars = ax.bar(x_pos, bias_stats['mean'], 
                     color=colors, edgecolor='black', alpha=0.7)
        max_height = bias_stats['mean'].max()
        min_height = bias_stats['mean'].min()
        y_range = max_height - min_height if max_height != min_height else 1
        y_margin = max(0.3, y_range * 0.15)  # 15% margin or at least 0.3
        
        for i, (bar, mean_val) in enumerate(zip(bars, bias_stats['mean'])):
            height = bar.get_height()
            text_y = height + y_margin * 0.3 if height >= 0 else height - y_margin * 0.3
            ax.text(bar.get_x() + bar.get_width()/2., text_y,
                   f'{mean_val:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bias_stats['score_state_90'], fontsize=11)
        ax.set_ylim(min_height - y_margin, max_height + y_margin)
    else:
        ax.set_ylim(-1, 1)  # Default range if no data
    ax.set_ylabel('Mean Excess Time (minutes)', fontsize=12)
    ax.set_title('Raw Targets: ALL Matches\n(n={}, excludes None values)'.format(len(df_raw)), 
                fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 5. Raw targets - Ghost games ONLY
    ax = axes[1]
    df_raw_ghost = df[(df['raw_excess_time_90'].notna()) & (df['is_ghost_game'] != False)]  # Includes True and None
    if len(df_raw_ghost) > 0:
        bias_stats = df_raw_ghost.groupby('score_state_90')['raw_excess_time_90'].agg(['mean', 'std']).reset_index()
        bias_stats = bias_stats.set_index('score_state_90').reindex(order).reset_index()
        x_pos = np.arange(len(bias_stats))
        
        bars = ax.bar(x_pos, bias_stats['mean'], 
                     color=colors, edgecolor='black', alpha=0.7)
        max_height = bias_stats['mean'].max()
        min_height = bias_stats['mean'].min()
        y_range = max_height - min_height if max_height != min_height else 1
        y_margin = max(0.3, y_range * 0.15)  # 15% margin or at least 0.3
        
        for i, (bar, mean_val) in enumerate(zip(bars, bias_stats['mean'])):
            height = bar.get_height()
            text_y = height + y_margin * 0.3 if height >= 0 else height - y_margin * 0.3
            ax.text(bar.get_x() + bar.get_width()/2., text_y,
                   f'{mean_val:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bias_stats['score_state_90'], fontsize=11)
        ax.set_ylim(min_height - y_margin, max_height + y_margin)
    else:
        ax.set_ylim(-1, 1)  # Default range if no data
    ax.set_ylabel('Mean Excess Time (minutes)', fontsize=12)
    ax.set_title('Raw Targets: Ghost Games ONLY\n(n={}, excludes None values)'.format(len(df_raw_ghost)), 
                fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 6. Raw targets - Non-ghost games ONLY
    ax = axes[2]
    df_raw_no_ghost = df[(df['raw_excess_time_90'].notna()) & (df['is_ghost_game'] == False)]  # Only False
    if len(df_raw_no_ghost) > 0:
        bias_stats = df_raw_no_ghost.groupby('score_state_90')['raw_excess_time_90'].agg(['mean', 'std']).reset_index()
        bias_stats = bias_stats.set_index('score_state_90').reindex(order).reset_index()
        x_pos = np.arange(len(bias_stats))
        
        bars = ax.bar(x_pos, bias_stats['mean'], 
                     color=colors, edgecolor='black', alpha=0.7)
        max_height = bias_stats['mean'].max()
        min_height = bias_stats['mean'].min()
        y_range = max_height - min_height if max_height != min_height else 1
        y_margin = max(0.3, y_range * 0.15)  # 15% margin or at least 0.3
        
        for i, (bar, mean_val) in enumerate(zip(bars, bias_stats['mean'])):
            height = bar.get_height()
            text_y = height + y_margin * 0.3 if height >= 0 else height - y_margin * 0.3
            ax.text(bar.get_x() + bar.get_width()/2., text_y,
                   f'{mean_val:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bias_stats['score_state_90'], fontsize=11)
        ax.set_ylim(min_height - y_margin, max_height + y_margin)
    else:
        ax.set_ylim(-1, 1)  # Default range if no data
    ax.set_ylabel('Mean Excess Time (minutes)', fontsize=12)
    ax.set_title('Raw Targets: Non-Ghost Games ONLY\n(n={}, excludes None values)'.format(len(df_raw_no_ghost)), 
                fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'home_bias_raw.png', bbox_inches='tight')
    plt.close()
    
    logger.info("Saved 2 home bias plots: home_bias_processed.png and home_bias_raw.png")


def plot_attendance_vs_excess(df: pd.DataFrame, output_dir: Path):
    """
    Plot attendance vs excess time, colored by score state.
    Excess time = actual_90 - announced_90. Positive = referee played longer than announced.
    Negative = referee ended early (rare, may indicate data issues or early whistle).
    
    Args:
        df: DataFrame with match data
        output_dir: Directory to save plots
    """
    logger.info("Plotting attendance vs excess")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color by score state
    for state, color in [('Home Leading', 'green'), ('Draw', 'gray'), ('Away Leading', 'red')]:
        subset = df[df['score_state_90'] == state]
        ax.scatter(subset['attendance'], subset['excess_time_90'], 
                  alpha=0.5, label=state, color=color)
    
    ax.set_xlabel('Attendance', fontsize=12)
    ax.set_ylabel('Excess Time (minutes)', fontsize=12)
    ax.set_title('Attendance vs Excess Time (colored by Score State)\n'
                'Excess Time = actual_90 - announced_90. Positive = played longer. Negative = ended early (rare).', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'attendance_vs_excess.png', bbox_inches='tight')
    plt.close()
    
    logger.info("Saved attendance_vs_excess.png")


def plot_raw_vs_processed_targets(df: pd.DataFrame, output_dir: Path):
    """
    Plot comparison of raw vs processed targets using difference analysis.
    
    Args:
        df: DataFrame with match data
        output_dir: Directory to save plots
    """
    logger.info("Plotting raw vs processed targets")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Calculate differences where both are available
    # Announced 45: Difference distribution
    ax1 = axes[0, 0]
    valid_data = df[df['raw_announced_time_45'].notna() & df['announced_45'].notna()].copy()
    if len(valid_data) > 0:
        valid_data['diff'] = valid_data['announced_45'] - valid_data['raw_announced_time_45']
        ax1.hist(valid_data['diff'], bins=range(int(valid_data['diff'].min()), int(valid_data['diff'].max()) + 2),
                color='steelblue', edgecolor='black')
        ax1.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Difference (Processed - Raw)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Announced 45: Processing Difference\n(0 = no change)', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
    
    # Announced 90: Difference distribution
    ax2 = axes[0, 1]
    valid_data = df[df['raw_announced_time_90'].notna() & df['announced_90'].notna()].copy()
    if len(valid_data) > 0:
        valid_data['diff'] = valid_data['announced_90'] - valid_data['raw_announced_time_90']
        ax2.hist(valid_data['diff'], bins=range(int(valid_data['diff'].min()), int(valid_data['diff'].max()) + 2),
                color='coral', edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Difference (Processed - Raw)', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Announced 90: Processing Difference\n(0 = no change)', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
    
    # Actual 45: Difference distribution
    ax3 = axes[1, 0]
    valid_data = df[df['raw_actual_played_45'].notna() & df['actual_45'].notna()].copy()
    if len(valid_data) > 0:
        valid_data['diff'] = valid_data['actual_45'] - valid_data['raw_actual_played_45']
        ax3.hist(valid_data['diff'], bins=range(int(valid_data['diff'].min()), int(valid_data['diff'].max()) + 2),
                color='green', edgecolor='black')
        ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('Difference (Processed - Raw)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Actual 45: Processing Difference\n(0 = no change)', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
    
    # Actual 90: Difference distribution
    ax4 = axes[1, 1]
    valid_data = df[df['raw_actual_played_90'].notna() & df['actual_90'].notna()].copy()
    if len(valid_data) > 0:
        valid_data['diff'] = valid_data['actual_90'] - valid_data['raw_actual_played_90']
        ax4.hist(valid_data['diff'], bins=range(int(valid_data['diff'].min()), int(valid_data['diff'].max()) + 2),
                color='purple', edgecolor='black')
        ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax4.set_xlabel('Difference (Processed - Raw)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title('Actual 90: Processing Difference\n(0 = no change)', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'raw_vs_processed_targets.png', bbox_inches='tight')
    plt.close()
    
    logger.info("Saved raw_vs_processed_targets.png")


def plot_all_flags_analysis(df: pd.DataFrame, output_dir: Path):
    """
    Plot comprehensive analysis of all flag parameters per season.
    Split into two separate PNGs: one for 45-minute flags and one for 90-minute flags.
    
    Args:
        df: DataFrame with match data
        output_dir: Directory to save plots
    """
    logger.info("Plotting all flags analysis (split by 45 and 90)")
    
    # Calculate flag statistics per season
    flag_stats = df.groupby('season').agg({
        'is_inferred_zero_45': 'mean',
        'is_inferred_zero_90': 'mean',
        'target_missing_45': 'mean',
        'target_missing_90': 'mean',
        'is_imputed_actual_45': 'mean',
        'is_imputed_actual_90': 'mean',
        'is_imputed_announced_45': 'mean',
        'is_imputed_announced_90': 'mean'
    }).reset_index()
    
    # Convert to percentages
    for col in flag_stats.columns:
        if col != 'season':
            flag_stats[col] *= 100
    
    # ===== 45-MINUTE FLAGS =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    flags_45 = [
        ('is_inferred_zero_45', 'Inferred Zero 45\n(% where announced overtime was set to 0)', axes[0, 0], 'steelblue'),
        ('target_missing_45', 'Target Missing 45\n(% where announced time was missing from raw data)', axes[0, 1], 'orange'),
        ('is_imputed_actual_45', 'Imputed Actual 45\n(% where actual played time was imputed from announced)', axes[1, 0], 'lightblue'),
        ('is_imputed_announced_45', 'Imputed Announced 45\n(% where announced time was corrected due to negative excess)', axes[1, 1], 'lightgreen')
    ]
    
    for flag_col, title, ax, color in flags_45:
        if flag_col in flag_stats.columns:
            bars = ax.bar(flag_stats['season'], flag_stats[flag_col], color=color, edgecolor='black')
            # Add values above bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        ax.set_xlabel('Season', fontsize=11)
        ax.set_ylabel('%', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_flags_analysis_45.png', bbox_inches='tight')
    plt.close()
    
    logger.info("Saved all_flags_analysis_45.png")
    
    # ===== 90-MINUTE FLAGS =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    flags_90 = [
        ('is_inferred_zero_90', 'Inferred Zero 90\n(% where announced overtime was set to 0)', axes[0, 0], 'coral'),
        ('target_missing_90', 'Target Missing 90\n(% where announced time was missing from raw data)', axes[0, 1], 'red'),
        ('is_imputed_actual_90', 'Imputed Actual 90\n(% where actual played time was imputed from announced)', axes[1, 0], 'lightcoral'),
        ('is_imputed_announced_90', 'Imputed Announced 90\n(% where announced time was corrected due to negative excess)', axes[1, 1], 'lightyellow')
    ]
    
    for flag_col, title, ax, color in flags_90:
        if flag_col in flag_stats.columns:
            bars = ax.bar(flag_stats['season'], flag_stats[flag_col], color=color, edgecolor='black')
            # Add values above bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        ax.set_xlabel('Season', fontsize=11)
        ax.set_ylabel('%', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_flags_analysis_90.png', bbox_inches='tight')
    plt.close()
    
    logger.info("Saved all_flags_analysis_90.png")




def generate_summary_report(df: pd.DataFrame, output_dir: Path):
    """
    Generate comprehensive summary report.
    
    Args:
        df: DataFrame with match data
        output_dir: Directory to save report
    """
    logger.info("Generating summary report")
    
    report_path = output_dir.parent / 'tables' / 'descriptive_report.txt'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BUNDESLIGABERT DESCRIPTIVE STATISTICS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Basic statistics
        f.write("1. BASIC STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total matches processed: {len(df)}\n")
        f.write(f"Total seasons: {df['season'].nunique()}\n")
        f.write(f"Seasons: {', '.join(sorted(df['season'].unique()))}\n\n")
        
        # Imputation rates (new flag structure)
        f.write("2. IMPUTATION AND FLAG STATISTICS\n")
        f.write("-" * 80 + "\n")
        
        # Imputed actual
        imputed_actual_45_pct = (df['is_imputed_actual_45'].sum() / len(df)) * 100
        imputed_actual_90_pct = (df['is_imputed_actual_90'].sum() / len(df)) * 100
        f.write(f"Matches with is_imputed_actual_45 = True: {df['is_imputed_actual_45'].sum()} ({imputed_actual_45_pct:.2f}%)\n")
        f.write(f"Matches with is_imputed_actual_90 = True: {df['is_imputed_actual_90'].sum()} ({imputed_actual_90_pct:.2f}%)\n\n")
        
        # Imputed announced
        if 'is_imputed_announced_45' in df.columns:
            imputed_announced_45_pct = (df['is_imputed_announced_45'].sum() / len(df)) * 100
            f.write(f"Matches with is_imputed_announced_45 = True: {df['is_imputed_announced_45'].sum()} ({imputed_announced_45_pct:.2f}%)\n")
        if 'is_imputed_announced_90' in df.columns:
            imputed_announced_90_pct = (df['is_imputed_announced_90'].sum() / len(df)) * 100
            f.write(f"Matches with is_imputed_announced_90 = True: {df['is_imputed_announced_90'].sum()} ({imputed_announced_90_pct:.2f}%)\n")
        f.write("\n")
        
        # Inferred zero
        inferred_zero_45_pct = (df['is_inferred_zero_45'].sum() / len(df)) * 100
        inferred_zero_90_pct = (df['is_inferred_zero_90'].sum() / len(df)) * 100
        f.write(f"Matches with is_inferred_zero_45 = True: {df['is_inferred_zero_45'].sum()} ({inferred_zero_45_pct:.2f}%)\n")
        f.write(f"Matches with is_inferred_zero_90 = True: {df['is_inferred_zero_90'].sum()} ({inferred_zero_90_pct:.2f}%)\n\n")
        
        # Target missing
        target_missing_45_pct = (df['target_missing_45'].sum() / len(df)) * 100
        target_missing_90_pct = (df['target_missing_90'].sum() / len(df)) * 100
        f.write(f"Matches with target_missing_45 = True: {df['target_missing_45'].sum()} ({target_missing_45_pct:.2f}%)\n")
        f.write(f"Matches with target_missing_90 = True: {df['target_missing_90'].sum()} ({target_missing_90_pct:.2f}%)\n\n")
        
        # Missing data
        f.write("3. MISSING DATA\n")
        f.write("-" * 80 + "\n")
        missing_cutoff = df['missing_cutoff_markers'].notna().sum()
        invalid_timeline = (~df['score_timeline_valid']).sum()
        f.write(f"Matches with missing_cutoff_markers (NOT null): {missing_cutoff}\n")
        f.write(f"Matches with score_timeline_valid = False: {invalid_timeline}\n\n")
        
        # Ghost games
        f.write("4. GHOST GAMES\n")
        f.write("-" * 80 + "\n")
        # Count True values only (None and False are not counted)
        ghost_total = df['is_ghost_game'].sum()
        ghost_pct = (ghost_total / len(df)) * 100
        f.write(f"Total ghost games: {ghost_total} ({ghost_pct:.2f}%)\n")
        f.write("\nGhost games per season:\n")
        # Convert to numeric for aggregation (True=1, False/None=0 for sum)
        df_ghost_numeric = df.copy()
        df_ghost_numeric['is_ghost_game_numeric'] = df_ghost_numeric['is_ghost_game'].map({True: 1, False: 0, None: 0}).astype(float)
        ghost_by_season = df_ghost_numeric.groupby('season')['is_ghost_game_numeric'].agg(['sum', 'count'])
        for season, row in ghost_by_season.iterrows():
            pct = (row['sum'] / row['count']) * 100
            f.write(f"  {season}: {int(row['sum'])} / {int(row['count'])} ({pct:.2f}%)\n")
        f.write("\n")
        
        # Ghost games = None (missing attendance data)
        f.write("Ghost games = None (missing attendance data):\n")
        ghost_none_total = df['is_ghost_game'].isna().sum()
        ghost_none_pct = (ghost_none_total / len(df)) * 100
        f.write(f"Total matches with missing attendance: {ghost_none_total} ({ghost_none_pct:.2f}%)\n")
        f.write("\nMissing attendance per season:\n")
        ghost_none_by_season = df.groupby('season')['is_ghost_game'].apply(lambda x: x.isna().sum()).reset_index(name='none_count')
        ghost_none_by_season = ghost_none_by_season.merge(
            df.groupby('season').size().reset_index(name='total'),
            on='season'
        )
        for _, row in ghost_none_by_season.iterrows():
            pct = (row['none_count'] / row['total']) * 100
            f.write(f"  {row['season']}: {int(row['none_count'])} / {int(row['total'])} ({pct:.2f}%)\n")
        f.write("\n")
        
        # Correlation matrices - split by half
        f.write("5. CORRELATION MATRIX: announced_45 vs 1st Half Events\n")
        f.write("-" * 80 + "\n")
        f.write("(Using processed data, only matches with non-missing announced_45)\n")
        event_cols_45 = ['goals_1st', 'subs_1st', 'cards_1st', 'var_1st', 
                        'injuries_1st', 'disturbances_1st', 'announced_45']
        available_cols_45 = [col for col in event_cols_45 if col in df.columns]
        if 'announced_45' in available_cols_45:
            # Only use matches with valid announced_45 (excludes None values)
            df_valid_45 = df[df['announced_45'].notna()].copy()
            f.write(f"Matches with valid announced_45: {len(df_valid_45)} / {len(df)} ({len(df_valid_45)/len(df)*100:.1f}%)\n")
            
            corr_matrix_45 = df_valid_45[available_cols_45].corr()['announced_45'].drop('announced_45')
            for col, corr in corr_matrix_45.items():
                f.write(f"  {col}: {corr:.4f}\n")
        f.write("\n")
        
        f.write("6. CORRELATION MATRIX: announced_90 vs 2nd Half Events\n")
        f.write("-" * 80 + "\n")
        f.write("(Using processed data, only matches with non-missing announced_90)\n")
        event_cols_90 = ['goals_2nd', 'subs_2nd', 'cards_2nd', 'var_2nd', 
                        'injuries_2nd', 'disturbances_2nd', 'announced_90']
        available_cols_90 = [col for col in event_cols_90 if col in df.columns]
        if 'announced_90' in available_cols_90:
            # Only use matches with valid announced_90 (excludes None values)
            # Note: If announced_45 is missing but announced_90 is available, still use announced_90
            df_valid_90 = df[df['announced_90'].notna()].copy()
            f.write(f"Matches with valid announced_90: {len(df_valid_90)} / {len(df)} ({len(df_valid_90)/len(df)*100:.1f}%)\n")
            
            corr_matrix_90 = df_valid_90[available_cols_90].corr()['announced_90'].drop('announced_90')
            for col, corr in corr_matrix_90.items():
                f.write(f"  {col}: {corr:.4f}\n")
        f.write("\n")
        
        f.write("7. CORRELATION MATRIX: actual_90 vs 2nd Half Events\n")
        f.write("-" * 80 + "\n")
        f.write("(Using processed data, only matches with non-missing actual_90)\n")
        event_cols_actual_90 = ['goals_2nd', 'subs_2nd', 'cards_2nd', 'var_2nd', 
                               'injuries_2nd', 'disturbances_2nd', 'actual_90']
        available_cols_actual_90 = [col for col in event_cols_actual_90 if col in df.columns]
        if 'actual_90' in available_cols_actual_90:
            # Only use matches with valid actual_90 (excludes None values)
            # Note: If actual_45 is missing but actual_90 is available, still use actual_90
            df_valid_actual_90 = df[df['actual_90'].notna()].copy()
            f.write(f"Matches with valid actual_90: {len(df_valid_actual_90)} / {len(df)} ({len(df_valid_actual_90)/len(df)*100:.1f}%)\n")
            
            corr_matrix_actual_90 = df_valid_actual_90[available_cols_actual_90].corr()['actual_90'].drop('actual_90')
            for col, corr in corr_matrix_actual_90.items():
                f.write(f"  {col}: {corr:.4f}\n")
        f.write("\n")
        
        f.write("8. CORRELATION MATRIX: excess_time_90 vs OT Events\n")
        f.write("-" * 80 + "\n")
        ot_cols = ['ot_goals_90', 'ot_subs_90', 'ot_var_90', 'ot_cards_90', 
                  'ot_injuries_90', 'ot_disturbances_90', 'excess_time_90']
        available_ot_cols = [col for col in ot_cols if col in df.columns]
        corr_matrix_ot = df[available_ot_cols].corr()['excess_time_90'].drop('excess_time_90')
        for col, corr in corr_matrix_ot.items():
            f.write(f"  {col}: {corr:.4f}\n")
        f.write("\n")
        
        # Raw vs Processed targets comparison
        f.write("9. RAW VS PROCESSED TARGETS COMPARISON\n")
        f.write("-" * 80 + "\n")
        
        # Count non-null raw values
        raw_announced_45_count = df['raw_announced_time_45'].notna().sum()
        raw_announced_90_count = df['raw_announced_time_90'].notna().sum()
        raw_actual_45_count = df['raw_actual_played_45'].notna().sum()
        raw_actual_90_count = df['raw_actual_played_90'].notna().sum()
        
        f.write(f"Raw announced_time_45 available: {raw_announced_45_count} / {len(df)} ({raw_announced_45_count/len(df)*100:.2f}%)\n")
        f.write(f"Raw announced_time_90 available: {raw_announced_90_count} / {len(df)} ({raw_announced_90_count/len(df)*100:.2f}%)\n")
        f.write(f"Raw actual_played_45 available: {raw_actual_45_count} / {len(df)} ({raw_actual_45_count/len(df)*100:.2f}%)\n")
        f.write(f"Raw actual_played_90 available: {raw_actual_90_count} / {len(df)} ({raw_actual_90_count/len(df)*100:.2f}%)\n\n")
        
        # Compare where both raw and processed are available
        both_45 = df[df['raw_announced_time_45'].notna() & df['announced_45'].notna()]
        both_90 = df[df['raw_announced_time_90'].notna() & df['announced_90'].notna()]
        
        if len(both_45) > 0:
            matches_45 = (both_45['raw_announced_time_45'] == both_45['announced_45']).sum()
            f.write(f"Raw == Processed (announced_45): {matches_45} / {len(both_45)} ({matches_45/len(both_45)*100:.2f}%)\n")
        
        if len(both_90) > 0:
            matches_90 = (both_90['raw_announced_time_90'] == both_90['announced_90']).sum()
            f.write(f"Raw == Processed (announced_90): {matches_90} / {len(both_90)} ({matches_90/len(both_90)*100:.2f}%)\n")
        f.write("\n")
        
        # Summary statistics for events
        f.write("10. EVENT SUMMARY STATISTICS BY PHASE\n")
        f.write("-" * 80 + "\n")
        
        event_phases = {
            'Goals': ['goals_1st', 'goals_2nd', 'ot_goals_45', 'ot_goals_90'],
            'Subs': ['subs_1st', 'subs_2nd', 'ot_subs_45', 'ot_subs_90'],
            'Cards': ['cards_1st', 'cards_2nd', 'ot_cards_45', 'ot_cards_90'],
            'VAR': ['var_1st', 'var_2nd', 'ot_var_45', 'ot_var_90'],
            'Injuries': ['injuries_1st', 'injuries_2nd', 'ot_injuries_45', 'ot_injuries_90'],
            'Disturbances': ['disturbances_1st', 'disturbances_2nd', 'ot_disturbances_45', 'ot_disturbances_90']
        }
        
        for event_name, cols in event_phases.items():
            f.write(f"\n{event_name}:\n")
            for col in cols:
                if col in df.columns:
                    stats = df[col].describe()
                    f.write(f"  {col}:\n")
                    f.write(f"    Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}, Min: {stats['min']:.2f}, Max: {stats['max']:.2f}\n")
        
        # Matches with excess time > 3 minutes
        f.write("\n11. MATCHES WITH EXCESS TIME > 3 MINUTES\n")
        f.write("-" * 80 + "\n")
        high_excess = df[df['excess_time_90'] > 3].copy()
        if len(high_excess) > 0:
            high_excess = high_excess.sort_values(['season', 'excess_time_90'], ascending=[True, False])
            f.write(f"Total matches with excess_time_90 > 3 minutes: {len(high_excess)}\n\n")
            for idx, row in high_excess.iterrows():
                home = row.get('home', 'Unknown')
                away = row.get('away', 'Unknown')
                season = row.get('season', 'Unknown')
                excess = row.get('excess_time_90', 0)
                match_id = row.get('match_id', 'Unknown')
                f.write(f"  {season}: {home} vs {away} (Excess: {excess:.2f} min, Match ID: {match_id})\n")
        else:
            f.write("No matches found with excess_time_90 > 3 minutes.\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Saved descriptive_report.txt")


def main():
    """
    Main function to orchestrate all analyses.
    """
    logger.info("Starting descriptive statistics analysis")
    
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data' / 'processed'
    figures_dir = project_root / 'reports' / 'processed_data' / 'figures'
    tables_dir = project_root / 'reports' / 'processed_data' / 'tables'
    
    # Create output directories
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading processed data...")
    df = load_processed_data(data_dir)
    
    if df.empty:
        logger.error("No data loaded! Check data directory.")
        return
    
    logger.info(f"Loaded {len(df)} matches")
    
    # Part 1: Data Integrity & Imputation
    logger.info("Part 1: Data Integrity & Imputation")
    plot_all_flags_analysis(df, figures_dir)
    plot_missing_data_analysis(df, figures_dir)
    plot_matches_per_season(df, figures_dir)
    plot_ghost_games_analysis(df, figures_dir)
    plot_raw_vs_processed_targets(df, figures_dir)
    
    # Part 2: Target Variable Analysis
    logger.info("Part 2: Target Variable Analysis")
    plot_announced_time_distribution(df, figures_dir)
    plot_time_creep(df, figures_dir)
    
    # Part 3: Excess Time Analysis
    logger.info("Part 3: Excess Time Analysis")
    plot_excess_time_distribution(df, figures_dir)
    table_excess_time_stats(df, figures_dir)
    
    # Part 4: Driver Analysis
    logger.info("Part 4: Driver Analysis")
    plot_event_distributions_combined(df, figures_dir)
    
    # Part 5: Overtime Analysis
    logger.info("Part 5: Overtime Analysis")
    plot_overtime_chaos(df, figures_dir)
    plot_excess_vs_ot_events(df, figures_dir)
    
    # Part 6: Bias & Pressure
    logger.info("Part 6: Bias & Pressure")
    plot_home_bias_raw(df, figures_dir)
    plot_attendance_vs_excess(df, figures_dir)
    
    # Summary Report
    logger.info("Generating summary report")
    generate_summary_report(df, figures_dir)
    
    logger.info("Analysis complete!")
    logger.info(f"Figures saved to: {figures_dir}")
    logger.info(f"Tables saved to: {tables_dir}")


if __name__ == '__main__':
    main()

