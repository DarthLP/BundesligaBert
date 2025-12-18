"""
Regression Analysis Script for BundesligaBERT

This script performs econometric analysis on Bundesliga stoppage time data to test
hypotheses about crowd pressure effects on referee decisions. It implements three
regression models:

1. Baseline Model (2nd Half): Predicts announced stoppage time using game events
2. Excess Time Model (2nd Half): Analyzes excess time (actual - announced) to detect bias
3. Placebo Test (1st Half): Tests if pressure effects exist in 1st half (should be insignificant)

The script includes pressure interaction variables that capture crowd influence in close games,
where home teams are either chasing a goal or defending a lead.

Usage:
    # Run from project root
    python src/analysis/regression_analysis.py
    
    # Or as module
    python -m src.analysis.regression_analysis

Input:
    Processed JSON files from data/processed/season_{season}/match_{match_id}.json

Output:
    - Model summaries (reports/regression/tables/):
      * regression_baseline_90_summary.txt (Model 1: Baseline 2nd Half)
      * regression_excess_90_summary.txt (Model 2: Excess Time 2nd Half)
      * regression_placebo_baseline_45_summary.txt (Model 3A: Placebo Baseline 1st Half)
      * regression_placebo_excess_45_summary.txt (Model 3B: Placebo Excess 1st Half)
    - Comparison metrics: reports/regression/baseline_comparison_data.json
    - Residuals CSV: reports/regression/tables/residuals_*.csv (one per model)
    - Visualizations (reports/regression/figures/):
      * plot_bias_comparison.png (Pressure coefficient comparison)
      * plot_predicted_vs_actual.png (Scatter plots for all models)

Author: BundesligaBERT Project
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    pass  # seaborn is optional, only used for styling

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plot parameters for publication-quality plots
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.style.use('default')  # Use default style if seaborn not available


def extract_score_at_minute(score_timeline: Dict[str, List[int]], minute: int, inclusive: bool = True) -> Tuple[int, int]:
    """
    Extract score at a specific minute from score timeline.
    
    Args:
        score_timeline: Dictionary mapping minute strings (e.g., "1", "45", "90") to [home_score, away_score]
        minute: Target minute to get score for
        inclusive: If True, includes scores at exactly the minute. If False, gets last score before the minute.
        
    Returns:
        Tuple of (home_score, away_score), defaulting to (0, 0) if no score found
    """
    if not score_timeline:
        return (0, 0)
    
    valid_scores = []
    
    for key, score in score_timeline.items():
        try:
            key_minute = int(key)
            if inclusive:
                if key_minute <= minute:
                    valid_scores.append((key_minute, score))
            else:
                if key_minute < minute:
                    valid_scores.append((key_minute, score))
        except (ValueError, TypeError):
            continue
    
    if valid_scores:
        # Get the last score (latest minute)
        latest_minute, latest_score = max(valid_scores, key=lambda x: x[0])
        return tuple(latest_score)
    else:
        return (0, 0)


def load_and_flatten_data(data_dir: Path, train_seasons: List[str], test_seasons: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load processed JSON files directly from season directories and flatten into DataFrames.
    
    Loads individual match files from data/processed/season_*/ directories, flattens nested
    structures, and extracts scores at 45 minutes from score_timeline. Separates into
    train and test DataFrames based on season.
    
    Args:
        data_dir: Path to data/processed directory
        train_seasons: List of season strings for training (e.g., ["2017-18", "2018-19", ...])
        test_seasons: List of season strings for testing (e.g., ["2024-25"])
        
    Returns:
        Tuple of (train_df, test_df) with flattened match data
    """
    logger.info("Loading and flattening data from individual JSON files")
    
    def flatten_match(match_data: Dict) -> Dict:
        """Flatten a single match JSON into a flat dictionary."""
        flat_match = {}
        
        # Basic fields
        flat_match['match_id'] = match_data.get('match_id')
        flat_match['season'] = match_data.get('season')
        
        # Flatten metadata
        metadata = match_data.get('metadata', {})
        flat_match['home'] = metadata.get('home')
        flat_match['away'] = metadata.get('away')
        flat_match['attendance'] = metadata.get('attendance')
        flat_match['matchday'] = metadata.get('matchday')
        
        # Flatten flags
        flags = match_data.get('flags', {})
        flat_match['is_imputed_actual_45'] = flags.get('is_imputed_actual_45', False)
        flat_match['is_imputed_actual_90'] = flags.get('is_imputed_actual_90', False)
        flat_match['is_imputed_announced_45'] = flags.get('is_imputed_announced_45', False)
        flat_match['is_imputed_announced_90'] = flags.get('is_imputed_announced_90', False)
        flat_match['target_missing_45'] = flags.get('target_missing_45', False)
        flat_match['target_missing_90'] = flags.get('target_missing_90', False)
        
        # Flatten targets
        targets = match_data.get('targets', {})
        flat_match['announced_45'] = targets.get('announced_45')
        flat_match['actual_45'] = targets.get('actual_45')
        flat_match['excess_45'] = targets.get('excess_45')
        flat_match['announced_90'] = targets.get('announced_90')
        flat_match['actual_90'] = targets.get('actual_90')
        flat_match['excess_90'] = targets.get('excess_90')
        
        # Flatten features_regular
        features_regular = match_data.get('features_regular', {})
        for key, value in features_regular.items():
            flat_match[key] = value
        
        # Flatten features_overtime
        features_overtime = match_data.get('features_overtime', {})
        for key, value in features_overtime.items():
            flat_match[key] = value
        
        # Extract scores at 45 minutes from score_timeline
        score_timeline = match_data.get('score_timeline', {})
        home_score_45, away_score_45 = extract_score_at_minute(score_timeline, 45, inclusive=True)
        flat_match['home_score_45'] = home_score_45
        flat_match['away_score_45'] = away_score_45
        
        return flat_match
    
    # Load matches from season directories
    train_matches = []
    test_matches = []
    
    # Find all season directories
    season_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('season_')])
    
    for season_dir in season_dirs:
        season = season_dir.name.replace('season_', '')
        logger.info(f"Processing {season_dir.name}")
        match_files = sorted(season_dir.glob('match_*.json'))
        
        for match_file in match_files:
            try:
                with open(match_file, 'r', encoding='utf-8') as f:
                    match_data = json.load(f)
                
                flat_match = flatten_match(match_data)
                
                if season in train_seasons:
                    train_matches.append(flat_match)
                elif season in test_seasons:
                    test_matches.append(flat_match)
                    
            except Exception as e:
                logger.warning(f"Failed to load {match_file}: {e}")
                continue
    
    # Create DataFrames
    train_df = pd.DataFrame(train_matches)
    test_df = pd.DataFrame(test_matches)
    
    # Force Majeure Filter: Exclude matches with excess_time_90 > 4 minutes
    # These represent structural breaks (riots, VAR confusion, technical failures, severe injuries)
    # rather than normal referee bias or time-wasting, and must be excluded to preserve statistical validity
    # See: src/data/remove_force_majeure_matches.py for detailed rationale
    if 'excess_90' in train_df.columns:
        train_before = len(train_df)
        train_df = train_df[train_df['excess_90'] <= 4.0].copy()
        train_removed = train_before - len(train_df)
        if train_removed > 0:
            logger.info(f"Removed {train_removed} force majeure matches from train set (excess_90 > 4.0 min)")
    
    if 'excess_90' in test_df.columns:
        test_before = len(test_df)
        test_df = test_df[test_df['excess_90'] <= 4.0].copy()
        test_removed = test_before - len(test_df)
        if test_removed > 0:
            logger.info(f"Removed {test_removed} force majeure matches from test set (excess_90 > 4.0 min)")
    
    logger.info(f"Train DataFrame shape: {train_df.shape} (seasons: {train_seasons})")
    logger.info(f"Test DataFrame shape: {test_df.shape} (seasons: {test_seasons})")
    
    return train_df, test_df


def filter_dataframes(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create 4 filtered DataFrames for different regression models.
    
    Args:
        df: Input DataFrame with all matches
        
    Returns:
        Tuple of (df_baseline_90, df_excess_90, df_placebo_45, df_excess_45)
    """
    logger.info("Filtering dataframes")
    
    # 1. df_baseline_90: Filter where announced_90 is not None/null (numeric)
    df_baseline_90 = df[df['announced_90'].notna()].copy()
    logger.info(f"df_baseline_90: {len(df_baseline_90)} matches")
    
    # 2. df_excess_90: Filter where target_missing_90 == False AND is_imputed_actual_90 == False
    # Also exclude force majeure matches (excess_90 > 4.0 min) - structural breaks, not bias
    df_excess_90 = df[
        (df['target_missing_90'] == False) & 
        (df['is_imputed_actual_90'] == False) &
        (df['excess_90'].isna() | (df['excess_90'] <= 4.0))
    ].copy()
    logger.info(f"df_excess_90: {len(df_excess_90)} matches (excluded force majeure: excess_90 > 4.0 min)")
    
    # 3. df_placebo_45: Filter where announced_45 is not None/null (numeric)
    df_placebo_45 = df[df['announced_45'].notna()].copy()
    logger.info(f"df_placebo_45: {len(df_placebo_45)} matches")
    
    # 4. df_excess_45: Filter where target_missing_45 == False AND is_imputed_actual_45 == False
    df_excess_45 = df[
        (df['target_missing_45'] == False) & 
        (df['is_imputed_actual_45'] == False)
    ].copy()
    logger.info(f"df_excess_45: {len(df_excess_45)} matches")
    
    return df_baseline_90, df_excess_90, df_placebo_45, df_excess_45


def engineer_pressure_features(df: pd.DataFrame, minute: int = 90) -> pd.DataFrame:
    """
    Engineer pressure interaction variables for regression analysis.
    
    Creates game state variables, pressure directions, and interaction terms
    that capture crowd influence in close games.
    
    Args:
        df: Input DataFrame
        minute: Either 90 (for 2nd half) or 45 (for 1st half)
        
    Returns:
        DataFrame with added pressure features
    """
    suffix = str(minute)
    df = df.copy()  # Avoid modifying original
    
    # Step A: Game States
    score_diff_col = f'score_diff_{suffix}'
    home_score_col = f'home_score_{suffix}'
    away_score_col = f'away_score_{suffix}'
    
    # Check if score columns exist, fill with 0 if missing
    if home_score_col not in df.columns:
        df[home_score_col] = 0
    if away_score_col not in df.columns:
        df[away_score_col] = 0
    
    # Fill NaN values with 0
    df[home_score_col] = df[home_score_col].fillna(0)
    df[away_score_col] = df[away_score_col].fillna(0)
    
    df[score_diff_col] = df[home_score_col] - df[away_score_col]
    df[f'is_close_game_{suffix}'] = df[score_diff_col].abs() <= 1
    df[f'is_blowout_{suffix}'] = df[score_diff_col].abs() >= 2
    
    # Step B: Pressure Directions (Split home_chasing into losing and draw)
    # home_losing_1: 1 if score_diff == -1 (home team down by 1), else 0
    df[f'home_losing_1_{suffix}'] = (df[score_diff_col] == -1).astype(int)
    # draw: 1 if score_diff == 0 (draw), else 0
    df[f'draw_{suffix}'] = (df[score_diff_col] == 0).astype(int)
    # home_defending: 1 if score_diff == 1 (home team up by 1), else 0
    df[f'home_defending_{suffix}'] = (df[score_diff_col] == 1).astype(int)
    
    # Step C: Interaction Terms
    # Normalize attendance (divide by 50000)
    # Handle missing attendance with season-specific imputation:
    # - COVID seasons (2019-20, 2020-21, 2021-22): Missing → 0 (ghost games, no crowd)
    # - Non-COVID seasons: Missing → season-specific median (data collection issue)
    if 'attendance' not in df.columns:
        df['attendance'] = None
    
    # Season-specific median attendance for non-COVID seasons (from descriptive analysis)
    season_medians = {
        '2017-18': 43250,
        '2018-19': 39100,
        '2022-23': 45147,
        '2023-24': 33305,
        '2024-25': 33305  # Use 2023-24 median as proxy for 2024-25
    }
    corona_seasons = ['2019-20', '2020-21', '2021-22']
    
    # Impute missing attendance based on season (vectorized approach)
    if 'season' in df.columns:
        # Identify rows with missing attendance (NaN/None only - 0 is a valid attendance value!)
        missing_mask = df['attendance'].isna()
        
        if missing_mask.any():
            # Create a mapping function for imputation values
            def get_imputation_value(season):
                if season in corona_seasons:
                    return 0  # COVID season: missing attendance → 0 (ghost games, no crowd)
                elif season in season_medians:
                    return season_medians[season]  # Non-COVID season: use season-specific median
                else:
                    return 36618  # Fallback: overall median
            
            # Apply imputation using vectorized map
            df.loc[missing_mask, 'attendance'] = df.loc[missing_mask, 'season'].map(get_imputation_value).astype(int)
    else:
        # If season column doesn't exist, use conservative approach: 0 for all missing
        df['attendance'] = df['attendance'].fillna(0)
    
    # Fill any remaining NaN values (shouldn't happen, but safety check)
    df['attendance'] = df['attendance'].fillna(0)
    
    df[f'attendance_norm_{suffix}'] = df['attendance'] / 50000
    
    # Pressure interaction terms
    # pressure_add: When home team is losing by 1 in close game (desperate, wants MORE time)
    df[f'pressure_add_{suffix}'] = (
        df[f'attendance_norm_{suffix}'] * 
        df[f'home_losing_1_{suffix}'] * 
        df[f'is_close_game_{suffix}'].astype(int)
    )
    
    # pressure_draw: When game is tied in close game (cautious, may waste time)
    df[f'pressure_draw_{suffix}'] = (
        df[f'attendance_norm_{suffix}'] * 
        df[f'draw_{suffix}'] * 
        df[f'is_close_game_{suffix}'].astype(int)
    )
    
    # pressure_end: When home team is defending 1-goal lead in close game (wants LESS time)
    df[f'pressure_end_{suffix}'] = (
        df[f'attendance_norm_{suffix}'] * 
        df[f'home_defending_{suffix}'] * 
        df[f'is_close_game_{suffix}'].astype(int)
    )
    
    return df


def convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python types
    """
    # Handle numpy scalar types (compatible with NumPy 1.x and 2.x)
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    # Also check for specific numpy dtypes
    elif type(obj).__module__ == 'numpy' and isinstance(obj, (np.int8, np.int16, np.int32, np.int64, 
                                                               np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif type(obj).__module__ == 'numpy' and isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj


def run_baseline_model_90(df_train: pd.DataFrame, df_test: pd.DataFrame, output_dir: Path) -> Dict:
    """
    Run Model 1: Main Baseline (2nd Half) regression.
    
    Predicts announced_90 using game events and pressure variables.
    
    Args:
        df_train: Training DataFrame (2017-18 to 2023-24)
        df_test: Test DataFrame (2024-25)
        output_dir: Directory to save model summary
        
    Returns:
        Dictionary with model results and metrics
    """
    logger.info("Running Model 1: Baseline 2nd Half")
    
    # Filter training data
    df_train_filtered = df_train[df_train['announced_90'].notna()].copy()
    
    # Handle empty test DataFrame
    if len(df_test) == 0 or 'announced_90' not in df_test.columns:
        df_test_filtered = pd.DataFrame()
        logger.warning("Test DataFrame is empty or missing columns. Skipping test set evaluation.")
    else:
        df_test_filtered = df_test[df_test['announced_90'].notna()].copy()
    
    # Engineer pressure features
    df_train_filtered = engineer_pressure_features(df_train_filtered, minute=90)
    if len(df_test_filtered) > 0:
        df_test_filtered = engineer_pressure_features(df_test_filtered, minute=90)
    
    # Check for required columns
    required_cols = ['goals_2nd', 'subs_2nd', 'cards_2nd', 'var_2nd', 'injuries_2nd', 
                     'disturbances_2nd', 'pressure_add_90', 'pressure_draw_90', 'pressure_end_90', 
                     'season', 'announced_90']
    missing_cols = [col for col in required_cols if col not in df_train_filtered.columns]
    if missing_cols:
        logger.warning(f"Missing columns in training data: {missing_cols}")
        # Fill missing columns with 0
        for col in missing_cols:
            if col != 'season':
                df_train_filtered[col] = 0
                if col in df_test_filtered.columns:
                    df_test_filtered[col] = df_test_filtered[col].fillna(0)
    
    # Prepare formula (removed is_blowout_90 to avoid multicollinearity with close_game)
    formula = (
        "announced_90 ~ goals_2nd + subs_2nd + cards_2nd + var_2nd + "
        "injuries_2nd + disturbances_2nd + pressure_add_90 + pressure_draw_90 + pressure_end_90 + "
        "C(season)"
    )
    
    # Fit model
    try:
        model = ols(formula, data=df_train_filtered).fit()
    except Exception as e:
        logger.error(f"Failed to fit model: {e}")
        raise
    
    # Predict on test set (if available)
    if len(df_test_filtered) > 0:
        df_test_filtered = df_test_filtered.dropna(subset=[
            'goals_2nd', 'subs_2nd', 'cards_2nd', 'var_2nd', 'injuries_2nd', 
            'disturbances_2nd', 'pressure_add_90', 'pressure_draw_90', 'pressure_end_90', 
            'season'
        ])
    
    if len(df_test_filtered) > 0:
        try:
            # Fix for unseen season (2024-25): Temporarily set to 2023-24 to use correct season coefficient
            # This assumes the "Net Playing Time" trend continues from 2023-24 to 2024-25
            df_test_predict = df_test_filtered.copy()
            season_2024_mask = df_test_predict['season'] == '2024-25'
            
            if season_2024_mask.sum() > 0:
                logger.info(f"Found {season_2024_mask.sum()} matches from 2024-25 season")
                logger.info("Temporarily setting season to 2023-24 for prediction (assuming trend continues)")
                # Store original season for potential restoration
                original_seasons = df_test_predict['season'].copy()
                # Set 2024-25 to 2023-24 for prediction
                df_test_predict.loc[season_2024_mask, 'season'] = '2023-24'
            else:
                original_seasons = None
            
            predictions = model.predict(df_test_predict)
            actual = df_test_filtered['announced_90']
            rmse = np.sqrt(np.mean((predictions - actual) ** 2))
            logger.info(f"Test RMSE: {rmse:.4f}")
            
            # Restore original season labels if we changed them
            if original_seasons is not None:
                df_test_filtered['season'] = original_seasons
                
        except Exception as e:
            rmse = None
            logger.warning(f"Could not calculate test RMSE: {e}")
    else:
        rmse = None
        logger.warning("No test data available for RMSE calculation")
    
    # Calculate additional metrics
    train_predictions = model.predict(df_train_filtered)
    train_actual = df_train_filtered['announced_90']
    train_r2_adj = model.rsquared_adj
    aic = model.aic
    
    if len(df_test_filtered) > 0 and rmse is not None:
        test_mae = np.mean(np.abs(predictions - actual))
    else:
        test_mae = None
    
    # Save summary
    output_file = output_dir / "regression_baseline_90_summary.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Model 1: Main Baseline (2nd Half)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Formula: {formula}\n\n")
        f.write(f"Training samples: {len(df_train_filtered)}\n")
        f.write(f"Test samples: {len(df_test_filtered)}\n")
        if rmse is not None:
            f.write(f"Test RMSE: {rmse:.4f}\n")
        if test_mae is not None:
            f.write(f"Test MAE: {test_mae:.4f}\n")
        f.write(f"Train R² Adjusted: {train_r2_adj:.4f}\n")
        f.write(f"AIC: {aic:.4f}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("Model Summary:\n")
        f.write("=" * 80 + "\n\n")
        f.write(str(model.summary()))
    
    logger.info(f"Saved model summary to {output_file}")
    
    # Save residuals
    residuals_file = output_dir / "residuals_baseline_90.csv"
    train_residuals = pd.DataFrame({
        'match_id': df_train_filtered['match_id'].values,
        'actual': train_actual.values,
        'predicted': train_predictions.values,
        'residual': (train_actual - train_predictions).values,
        'is_test_set': False
    })
    
    if len(df_test_filtered) > 0 and 'predictions' in locals():
        test_residuals = pd.DataFrame({
            'match_id': df_test_filtered['match_id'].values,
            'actual': actual.values,
            'predicted': predictions.values,
            'residual': (actual - predictions).values,
            'is_test_set': True
        })
        all_residuals = pd.concat([train_residuals, test_residuals], ignore_index=True)
    else:
        all_residuals = train_residuals
    
    all_residuals.to_csv(residuals_file, index=False)
    logger.info(f"Saved residuals to {residuals_file}")
    
    # Extract coefficients and p-values
    coefficients = {param: float(model.params[param]) for param in model.params.index}
    p_values = {param: float(model.pvalues[param]) for param in model.pvalues.index}
    
    # Prepare return dictionary
    result = {
        'model': model,
        'rmse': rmse,
        'test_mae': test_mae,
        'train_r2_adj': train_r2_adj,
        'aic': aic,
        'n_train': len(df_train_filtered),
        'n_test': len(df_test_filtered),
        'coefficients': coefficients,
        'p_values': p_values,
        'train_df': df_train_filtered,
        'test_df': df_test_filtered,
        'train_predictions': train_predictions
    }
    
    # Add test predictions if available
    if len(df_test_filtered) > 0 and 'predictions' in locals():
        result['test_predictions'] = predictions
    else:
        result['test_predictions'] = None
    
    return result


def run_excess_model_90(df: pd.DataFrame, output_dir: Path) -> Dict:
    """
    Run Model 2: Excess Time (2nd Half) regression.
    
    Analyzes excess_90 (actual - announced) to detect bias from crowd pressure.
    
    Args:
        df: Combined DataFrame (train + test) with filtered excess_90 data
        output_dir: Directory to save model summary
        
    Returns:
        Dictionary with model results
    """
    logger.info("Running Model 2: Excess Time 2nd Half")
    
    # Filter data
    df_filtered = df[
        (df['target_missing_90'] == False) & 
        (df['is_imputed_actual_90'] == False) &
        (df['excess_90'].notna())
    ].copy()
    
    # Engineer pressure features
    df_filtered = engineer_pressure_features(df_filtered, minute=90)
    
    # Prepare formula (removed is_blowout_90 to avoid multicollinearity with close_game)
    formula = (
        "excess_90 ~ ot_goals_90 + ot_subs_90 + ot_cards_90 + ot_var_90 + "
        "ot_injuries_90 + ot_disturbances_90 + pressure_add_90 + pressure_draw_90 + pressure_end_90"
    )
    
    # Fit model
    model = ols(formula, data=df_filtered).fit()
    
    # Calculate metrics
    train_r2_adj = model.rsquared_adj
    aic = model.aic
    predictions = model.predict(df_filtered)
    actual = df_filtered['excess_90']
    
    # Save summary
    output_file = output_dir / "regression_excess_90_summary.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Model 2: Excess Time (2nd Half)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Formula: {formula}\n\n")
        f.write(f"Samples: {len(df_filtered)}\n")
        f.write(f"Train R² Adjusted: {train_r2_adj:.4f}\n")
        f.write(f"AIC: {aic:.4f}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("Model Summary:\n")
        f.write("=" * 80 + "\n\n")
        f.write(str(model.summary()))
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("Pressure Coefficient Analysis:\n")
        f.write("=" * 80 + "\n\n")
        
        # Extract pressure coefficients
        if 'pressure_add_90' in model.params.index:
            coef_add = model.params['pressure_add_90']
            pvalue_add = model.pvalues['pressure_add_90']
            f.write(f"pressure_add_90 (Losing by 1) coefficient: {coef_add:.4f}\n")
            f.write(f"pressure_add_90 p-value: {pvalue_add:.4f}\n")
            f.write(f"pressure_add_90 significant: {'Yes' if pvalue_add < 0.05 else 'No'}\n\n")
        
        if 'pressure_draw_90' in model.params.index:
            coef_draw = model.params['pressure_draw_90']
            pvalue_draw = model.pvalues['pressure_draw_90']
            f.write(f"pressure_draw_90 (Draw) coefficient: {coef_draw:.4f}\n")
            f.write(f"pressure_draw_90 p-value: {pvalue_draw:.4f}\n")
            f.write(f"pressure_draw_90 significant: {'Yes' if pvalue_draw < 0.05 else 'No'}\n\n")
        
        if 'pressure_end_90' in model.params.index:
            coef_end = model.params['pressure_end_90']
            pvalue_end = model.pvalues['pressure_end_90']
            f.write(f"pressure_end_90 (Defending Lead) coefficient: {coef_end:.4f}\n")
            f.write(f"pressure_end_90 p-value: {pvalue_end:.4f}\n")
            f.write(f"pressure_end_90 significant: {'Yes' if pvalue_end < 0.05 else 'No'}\n")
    
    logger.info(f"Saved model summary to {output_file}")
    
    # Save residuals
    residuals_file = output_dir / "residuals_excess_90.csv"
    residuals_df = pd.DataFrame({
        'match_id': df_filtered['match_id'].values,
        'actual': actual.values,
        'predicted': predictions.values,
        'residual': (actual - predictions).values,
        'is_test_set': df_filtered['season'].isin(["2024-25"]).values
    })
    residuals_df.to_csv(residuals_file, index=False)
    logger.info(f"Saved residuals to {residuals_file}")
    
    # Extract coefficients and p-values
    coefficients = {param: float(model.params[param]) for param in model.params.index}
    p_values = {param: float(model.pvalues[param]) for param in model.pvalues.index}
    
    return {
        'model': model,
        'train_r2_adj': train_r2_adj,
        'aic': aic,
        'n_samples': len(df_filtered),
        'coefficients': coefficients,
        'p_values': p_values,
        'df': df_filtered,
        'predictions': predictions
    }


def run_placebo_test_45(df: pd.DataFrame, output_dir: Path) -> Dict:
    """
    Run Model 3: Placebo Test (1st Half) regression.
    
    Tests if pressure effects exist in 1st half (should be insignificant).
    Includes both baseline (announced_45) and excess time (excess_45) models.
    
    Args:
        df: Combined DataFrame (train + test) with filtered 1st half data
        output_dir: Directory to save model summary
        
    Returns:
        Dictionary with model results for both 3A and 3B
    """
    logger.info("Running Model 3: Placebo Test 1st Half")
    
    # Model 3A: Baseline 1st Half
    df_3a = df[df['announced_45'].notna()].copy()
    df_3a = engineer_pressure_features(df_3a, minute=45)
    
    formula_3a = (
        "announced_45 ~ goals_1st + subs_1st + cards_1st + var_1st + "
        "injuries_1st + disturbances_1st + pressure_add_45 + pressure_draw_45 + pressure_end_45 + "
        "C(season)"
    )
    
    model_3a = ols(formula_3a, data=df_3a).fit()
    
    # Model 3B: Excess Time 1st Half
    df_3b = df[
        (df['target_missing_45'] == False) & 
        (df['is_imputed_actual_45'] == False) &
        (df['excess_45'].notna())
    ].copy()
    df_3b = engineer_pressure_features(df_3b, minute=45)
    
    formula_3b = (
        "excess_45 ~ ot_goals_45 + ot_subs_45 + ot_cards_45 + ot_var_45 + "
        "ot_injuries_45 + ot_disturbances_45 + pressure_add_45 + pressure_draw_45 + pressure_end_45"
    )
    
    model_3b = ols(formula_3b, data=df_3b).fit()
    
    # Calculate metrics for both models
    train_r2_adj_3a = model_3a.rsquared_adj
    aic_3a = model_3a.aic
    predictions_3a = model_3a.predict(df_3a)
    actual_3a = df_3a['announced_45']
    
    train_r2_adj_3b = model_3b.rsquared_adj
    aic_3b = model_3b.aic
    predictions_3b = model_3b.predict(df_3b)
    actual_3b = df_3b['excess_45']
    
    # Save Model 3A summary
    output_file_3a = output_dir / "regression_placebo_baseline_45_summary.txt"
    output_file_3a.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file_3a, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Model 3A: Placebo Baseline 1st Half (announced_45)\n")
        f.write("=" * 80 + "\n\n")
        f.write("Hypothesis: Pressure coefficients should be INSIGNIFICANT (p > 0.05) in 1st half\n")
        f.write("If significant, this suggests the model might be flawed.\n\n")
        f.write(f"Formula: {formula_3a}\n\n")
        f.write(f"Samples: {len(df_3a)}\n")
        f.write(f"Train R² Adjusted: {train_r2_adj_3a:.4f}\n")
        f.write(f"AIC: {aic_3a:.4f}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("Model Summary:\n")
        f.write("=" * 80 + "\n\n")
        f.write(str(model_3a.summary()))
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("Pressure Coefficient Analysis:\n")
        f.write("=" * 80 + "\n\n")
        
        if 'pressure_add_45' in model_3a.params.index:
            coef_add = model_3a.params['pressure_add_45']
            pvalue_add = model_3a.pvalues['pressure_add_45']
            f.write(f"pressure_add_45 (Losing by 1) coefficient: {coef_add:.4f}\n")
            f.write(f"pressure_add_45 p-value: {pvalue_add:.4f}\n")
            f.write(f"pressure_add_45 significant: {'Yes' if pvalue_add < 0.05 else 'No'}\n\n")
        
        if 'pressure_draw_45' in model_3a.params.index:
            coef_draw = model_3a.params['pressure_draw_45']
            pvalue_draw = model_3a.pvalues['pressure_draw_45']
            f.write(f"pressure_draw_45 (Draw) coefficient: {coef_draw:.4f}\n")
            f.write(f"pressure_draw_45 p-value: {pvalue_draw:.4f}\n")
            f.write(f"pressure_draw_45 significant: {'Yes' if pvalue_draw < 0.05 else 'No'}\n\n")
        
        if 'pressure_end_45' in model_3a.params.index:
            coef_end = model_3a.params['pressure_end_45']
            pvalue_end = model_3a.pvalues['pressure_end_45']
            f.write(f"pressure_end_45 (Defending Lead) coefficient: {coef_end:.4f}\n")
            f.write(f"pressure_end_45 p-value: {pvalue_end:.4f}\n")
            f.write(f"pressure_end_45 significant: {'Yes' if pvalue_end < 0.05 else 'No'}\n")
    
    logger.info(f"Saved Model 3A summary to {output_file_3a}")
    
    # Save Model 3B summary
    output_file_3b = output_dir / "regression_placebo_excess_45_summary.txt"
    output_file_3b.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file_3b, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Model 3B: Placebo Excess Time 1st Half (excess_45)\n")
        f.write("=" * 80 + "\n\n")
        f.write("Hypothesis: Pressure coefficients should be INSIGNIFICANT (p > 0.05) in 1st half\n")
        f.write("If significant, this suggests the model might be flawed.\n\n")
        f.write(f"Formula: {formula_3b}\n\n")
        f.write(f"Samples: {len(df_3b)}\n")
        f.write(f"Train R² Adjusted: {train_r2_adj_3b:.4f}\n")
        f.write(f"AIC: {aic_3b:.4f}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("Model Summary:\n")
        f.write("=" * 80 + "\n\n")
        f.write(str(model_3b.summary()))
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("Pressure Coefficient Analysis:\n")
        f.write("=" * 80 + "\n\n")
        
        if 'pressure_add_45' in model_3b.params.index:
            coef_add = model_3b.params['pressure_add_45']
            pvalue_add = model_3b.pvalues['pressure_add_45']
            f.write(f"pressure_add_45 (Losing by 1) coefficient: {coef_add:.4f}\n")
            f.write(f"pressure_add_45 p-value: {pvalue_add:.4f}\n")
            f.write(f"pressure_add_45 significant: {'Yes' if pvalue_add < 0.05 else 'No'}\n\n")
        
        if 'pressure_draw_45' in model_3b.params.index:
            coef_draw = model_3b.params['pressure_draw_45']
            pvalue_draw = model_3b.pvalues['pressure_draw_45']
            f.write(f"pressure_draw_45 (Draw) coefficient: {coef_draw:.4f}\n")
            f.write(f"pressure_draw_45 p-value: {pvalue_draw:.4f}\n")
            f.write(f"pressure_draw_45 significant: {'Yes' if pvalue_draw < 0.05 else 'No'}\n\n")
        
        if 'pressure_end_45' in model_3b.params.index:
            coef_end = model_3b.params['pressure_end_45']
            pvalue_end = model_3b.pvalues['pressure_end_45']
            f.write(f"pressure_end_45 (Defending Lead) coefficient: {coef_end:.4f}\n")
            f.write(f"pressure_end_45 p-value: {pvalue_end:.4f}\n")
            f.write(f"pressure_end_45 significant: {'Yes' if pvalue_end < 0.05 else 'No'}\n")
    
    logger.info(f"Saved Model 3B summary to {output_file_3b}")
    
    # Save residuals for Model 3A
    residuals_file_3a = output_dir / "residuals_placebo_baseline_45.csv"
    residuals_3a = pd.DataFrame({
        'match_id': df_3a['match_id'].values,
        'actual': actual_3a.values,
        'predicted': predictions_3a.values,
        'residual': (actual_3a - predictions_3a).values,
        'is_test_set': df_3a['season'].isin(["2024-25"]).values
    })
    residuals_3a.to_csv(residuals_file_3a, index=False)
    logger.info(f"Saved Model 3A residuals to {residuals_file_3a}")
    
    # Save residuals for Model 3B
    residuals_file_3b = output_dir / "residuals_placebo_excess_45.csv"
    residuals_3b = pd.DataFrame({
        'match_id': df_3b['match_id'].values,
        'actual': actual_3b.values,
        'predicted': predictions_3b.values,
        'residual': (actual_3b - predictions_3b).values,
        'is_test_set': df_3b['season'].isin(["2024-25"]).values
    })
    residuals_3b.to_csv(residuals_file_3b, index=False)
    logger.info(f"Saved Model 3B residuals to {residuals_file_3b}")
    
    # Extract coefficients and p-values
    coefficients_3a = {param: float(model_3a.params[param]) for param in model_3a.params.index}
    p_values_3a = {param: float(model_3a.pvalues[param]) for param in model_3a.pvalues.index}
    
    coefficients_3b = {param: float(model_3b.params[param]) for param in model_3b.params.index}
    p_values_3b = {param: float(model_3b.pvalues[param]) for param in model_3b.pvalues.index}
    
    return {
        'model_3a': model_3a,
        'model_3b': model_3b,
        'train_r2_adj_3a': train_r2_adj_3a,
        'train_r2_adj_3b': train_r2_adj_3b,
        'aic_3a': aic_3a,
        'aic_3b': aic_3b,
        'n_samples_3a': len(df_3a),
        'n_samples_3b': len(df_3b),
        'coefficients_3a': coefficients_3a,
        'coefficients_3b': coefficients_3b,
        'p_values_3a': p_values_3a,
        'p_values_3b': p_values_3b,
        'df_3a': df_3a,
        'df_3b': df_3b,
        'predictions_3a': predictions_3a,
        'predictions_3b': predictions_3b
    }


def save_comparison_metrics(results: Dict, output_file: Path) -> None:
    """
    Save comparison metrics for all 4 models to JSON file.
    
    Args:
        results: Dictionary containing all model results
        output_file: Path to save JSON file
    """
    logger.info("Saving comparison metrics to JSON")
    
    model1 = results.get('model1', {})
    model2 = results.get('model2', {})
    model3 = results.get('model3', {})
    
    comparison_data = {
        "baseline_model_90": {
            "test_rmse": convert_numpy_types(model1.get('rmse')),
            "test_mae": convert_numpy_types(model1.get('test_mae')),
            "train_r2_adj": convert_numpy_types(model1.get('train_r2_adj')),
            "aic": convert_numpy_types(model1.get('aic')),
            "n_obs_train": int(model1.get('n_train', 0)),
            "n_obs_test": int(model1.get('n_test', 0)),
            "coefficients": convert_numpy_types(model1.get('coefficients', {})),
            "p_values": convert_numpy_types(model1.get('p_values', {}))
        },
        "excess_model_90": {
            "train_r2_adj": convert_numpy_types(model2.get('train_r2_adj')),
            "aic": convert_numpy_types(model2.get('aic')),
            "n_obs": int(model2.get('n_samples', 0)),
            "coefficients": convert_numpy_types(model2.get('coefficients', {})),
            "p_values": convert_numpy_types(model2.get('p_values', {}))
        },
        "placebo_baseline_model_45": {
            "train_r2_adj": convert_numpy_types(model3.get('train_r2_adj_3a')),
            "aic": convert_numpy_types(model3.get('aic_3a')),
            "n_obs": int(model3.get('n_samples_3a', 0)),
            "coefficients": convert_numpy_types(model3.get('coefficients_3a', {})),
            "p_values": convert_numpy_types(model3.get('p_values_3a', {}))
        },
        "placebo_excess_model_45": {
            "train_r2_adj": convert_numpy_types(model3.get('train_r2_adj_3b')),
            "aic": convert_numpy_types(model3.get('aic_3b')),
            "n_obs": int(model3.get('n_samples_3b', 0)),
            "coefficients": convert_numpy_types(model3.get('coefficients_3b', {})),
            "p_values": convert_numpy_types(model3.get('p_values_3b', {}))
        }
    }
    
    # Convert all numpy types to Python types
    comparison_data = convert_numpy_types(comparison_data)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved comparison metrics to {output_file}")


def plot_predicted_vs_actual(results: Dict, output_file: Path) -> None:
    """
    Create scatter plots of predicted vs actual for all models (Test Set only).
    
    Args:
        results: Dictionary containing all model results
        output_file: Path to save figure
    """
    logger.info("Creating predicted vs actual scatter plots")
    
    model1 = results.get('model1', {})
    model2 = results.get('model2', {})
    model3 = results.get('model3', {})
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Predicted vs Actual (Test Set)', fontsize=16, fontweight='bold')
    
    # Model 1: Baseline 90 (Test Set)
    ax1 = axes[0, 0]
    test_df_1 = model1.get('test_df')
    test_pred_1 = model1.get('test_predictions')
    if test_df_1 is not None and test_pred_1 is not None and len(test_df_1) > 0:
        actual_1 = test_df_1['announced_90'].values
        ax1.scatter(actual_1, test_pred_1, alpha=0.6, s=50)
        # Add diagonal line
        min_val = min(actual_1.min(), test_pred_1.min())
        max_val = max(actual_1.max(), test_pred_1.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title('Second Half Announced Time (Test Set)')
        ax1.legend()
        ax1.grid(alpha=0.3)
    
    # Model 2: Excess 90 (Test Set only)
    ax2 = axes[0, 1]
    df_2 = model2.get('df')
    pred_2 = model2.get('predictions')
    if df_2 is not None and pred_2 is not None:
        test_mask = df_2['season'].isin(["2024-25"])
        if test_mask.sum() > 0:
            actual_2 = df_2.loc[test_mask, 'excess_90'].values
            pred_2_test = pred_2[test_mask.values]
            ax2.scatter(actual_2, pred_2_test, alpha=0.6, s=50)
            min_val = min(actual_2.min(), pred_2_test.min())
            max_val = max(actual_2.max(), pred_2_test.max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            ax2.set_xlabel('Actual')
            ax2.set_ylabel('Predicted')
            ax2.set_title('Second Half Excess Time (Test Set)')
            ax2.legend()
            ax2.grid(alpha=0.3)
    
    # Model 3A: Placebo Baseline 45 (Test Set only)
    ax3 = axes[1, 0]
    df_3a = model3.get('df_3a')
    pred_3a = model3.get('predictions_3a')
    if df_3a is not None and pred_3a is not None:
        test_mask_3a = df_3a['season'].isin(["2024-25"])
        if test_mask_3a.sum() > 0:
            actual_3a = df_3a.loc[test_mask_3a, 'announced_45'].values
            pred_3a_test = pred_3a[test_mask_3a.values]
            ax3.scatter(actual_3a, pred_3a_test, alpha=0.6, s=50)
            min_val = min(actual_3a.min(), pred_3a_test.min())
            max_val = max(actual_3a.max(), pred_3a_test.max())
            ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            ax3.set_xlabel('Actual')
            ax3.set_ylabel('Predicted')
            ax3.set_title('First Half Announced Time (Test Set)')
            ax3.legend()
            ax3.grid(alpha=0.3)
    
    # Model 3B: Placebo Excess 45 (Test Set only)
    ax4 = axes[1, 1]
    df_3b = model3.get('df_3b')
    pred_3b = model3.get('predictions_3b')
    if df_3b is not None and pred_3b is not None:
        test_mask_3b = df_3b['season'].isin(["2024-25"])
        if test_mask_3b.sum() > 0:
            actual_3b = df_3b.loc[test_mask_3b, 'excess_45'].values
            pred_3b_test = pred_3b[test_mask_3b.values]
            ax4.scatter(actual_3b, pred_3b_test, alpha=0.6, s=50)
            min_val = min(actual_3b.min(), pred_3b_test.min())
            max_val = max(actual_3b.max(), pred_3b_test.max())
            ax4.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            ax4.set_xlabel('Actual')
            ax4.set_ylabel('Predicted')
            ax4.set_title('First Half Excess Time (Test Set)')
            ax4.legend()
            ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved predicted vs actual plot to {output_file}")


def plot_bias_comparison(results: Dict, output_dir: Path) -> None:
    """
    Create bar chart comparing pressure coefficients across models.
    
    Compares pressure_add and pressure_end coefficients between:
    - Model 1 (90min) vs Model 3A (45min)
    - Model 2 (90min) vs Model 3B (45min)
    
    Args:
        results: Dictionary containing all model results
        output_dir: Directory to save figure
    """
    logger.info("Creating bias comparison plot")
    
    # Extract coefficients and confidence intervals
    model1 = results.get('model1', {}).get('model')
    model2 = results.get('model2', {}).get('model')
    model3a = results.get('model3', {}).get('model_3a')
    model3b = results.get('model3', {}).get('model_3b')
    
    if not all([model1, model2, model3a, model3b]):
        logger.warning("Missing models for comparison plot. Skipping visualization.")
        return
    
    # Extract coefficients and confidence intervals
    def get_coef_ci(model, var_name):
        """Extract coefficient and 95% confidence interval."""
        if var_name not in model.params.index:
            return None, None, None
        
        coef = model.params[var_name]
        ci = model.conf_int().loc[var_name]
        ci_lower = ci[0]
        ci_upper = ci[1]
        return coef, ci_lower, ci_upper
    
    # Collect data for plotting
    plot_data = []
    
    # Model 1 vs Model 3A (pressure_add)
    coef1_add, ci1_lower_add, ci1_upper_add = get_coef_ci(model1, 'pressure_add_90')
    coef3a_add, ci3a_lower_add, ci3a_upper_add = get_coef_ci(model3a, 'pressure_add_45')
    
    if coef1_add is not None:
        plot_data.append({
            'Model': 'Second Half Announced Time',
            'Coefficient': 'Crowd Pressure: Extend Time',
            'Value': coef1_add,
            'CI_Lower': ci1_lower_add,
            'CI_Upper': ci1_upper_add
        })
    if coef3a_add is not None:
        plot_data.append({
            'Model': 'First Half Announced Time',
            'Coefficient': 'Crowd Pressure: Extend Time',
            'Value': coef3a_add,
            'CI_Lower': ci3a_lower_add,
            'CI_Upper': ci3a_upper_add
        })
    
    # Model 1 vs Model 3A (pressure_end)
    coef1_end, ci1_lower_end, ci1_upper_end = get_coef_ci(model1, 'pressure_end_90')
    coef3a_end, ci3a_lower_end, ci3a_upper_end = get_coef_ci(model3a, 'pressure_end_45')
    
    if coef1_end is not None:
        plot_data.append({
            'Model': 'Second Half Announced Time',
            'Coefficient': 'Crowd Pressure: End Game',
            'Value': coef1_end,
            'CI_Lower': ci1_lower_end,
            'CI_Upper': ci1_upper_end
        })
    if coef3a_end is not None:
        plot_data.append({
            'Model': 'First Half Announced Time',
            'Coefficient': 'Crowd Pressure: End Game',
            'Value': coef3a_end,
            'CI_Lower': ci3a_lower_end,
            'CI_Upper': ci3a_upper_end
        })
    
    # Model 2 vs Model 3B (pressure_add)
    coef2_add, ci2_lower_add, ci2_upper_add = get_coef_ci(model2, 'pressure_add_90')
    coef3b_add, ci3b_lower_add, ci3b_upper_add = get_coef_ci(model3b, 'pressure_add_45')
    
    if coef2_add is not None:
        plot_data.append({
            'Model': 'Second Half Excess Time',
            'Coefficient': 'Crowd Pressure: Extend Time',
            'Value': coef2_add,
            'CI_Lower': ci2_lower_add,
            'CI_Upper': ci2_upper_add
        })
    if coef3b_add is not None:
        plot_data.append({
            'Model': 'First Half Excess Time',
            'Coefficient': 'Crowd Pressure: Extend Time',
            'Value': coef3b_add,
            'CI_Lower': ci3b_lower_add,
            'CI_Upper': ci3b_upper_add
        })
    
    # Model 2 vs Model 3B (pressure_end)
    coef2_end, ci2_lower_end, ci2_upper_end = get_coef_ci(model2, 'pressure_end_90')
    coef3b_end, ci3b_lower_end, ci3b_upper_end = get_coef_ci(model3b, 'pressure_end_45')
    
    if coef2_end is not None:
        plot_data.append({
            'Model': 'Second Half Excess Time',
            'Coefficient': 'Crowd Pressure: End Game',
            'Value': coef2_end,
            'CI_Lower': ci2_lower_end,
            'CI_Upper': ci2_upper_end
        })
    if coef3b_end is not None:
        plot_data.append({
            'Model': 'First Half Excess Time',
            'Coefficient': 'Crowd Pressure: End Game',
            'Value': coef3b_end,
            'CI_Lower': ci3b_lower_end,
            'CI_Upper': ci3b_upper_end
        })
    
    if not plot_data:
        logger.warning("No data to plot. Skipping visualization.")
        return
    
    df_plot = pd.DataFrame(plot_data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Pressure Coefficient Comparison: 90min vs 45min (Placebo Test)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Announced Time models - Extend Time pressure
    ax1 = axes[0, 0]
    data_add_1 = df_plot[(df_plot['Coefficient'] == 'Crowd Pressure: Extend Time') & 
                         (df_plot['Model'].isin(['Second Half Announced Time', 'First Half Announced Time']))]
    if len(data_add_1) > 0:
        x_pos = range(len(data_add_1))
        bars = ax1.bar(x_pos, data_add_1['Value'], yerr=[
            data_add_1['Value'] - data_add_1['CI_Lower'],
            data_add_1['CI_Upper'] - data_add_1['Value']
        ], capsize=5, alpha=0.7)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(data_add_1['Model'], rotation=45, ha='right')
        ax1.set_ylabel('Coefficient Value')
        ax1.set_title('Announced Time: Crowd Pressure to Extend Time\n(Home Team Chasing in Close Game)')
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Announced Time models - End Game pressure
    ax2 = axes[0, 1]
    data_end_1 = df_plot[(df_plot['Coefficient'] == 'Crowd Pressure: End Game') & 
                         (df_plot['Model'].isin(['Second Half Announced Time', 'First Half Announced Time']))]
    if len(data_end_1) > 0:
        x_pos = range(len(data_end_1))
        bars = ax2.bar(x_pos, data_end_1['Value'], yerr=[
            data_end_1['Value'] - data_end_1['CI_Lower'],
            data_end_1['CI_Upper'] - data_end_1['Value']
        ], capsize=5, alpha=0.7)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(data_end_1['Model'], rotation=45, ha='right')
        ax2.set_ylabel('Coefficient Value')
        ax2.set_title('Announced Time: Crowd Pressure to End Game\n(Home Team Defending Lead in Close Game)')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Excess Time models - Extend Time pressure
    ax3 = axes[1, 0]
    data_add_2 = df_plot[(df_plot['Coefficient'] == 'Crowd Pressure: Extend Time') & 
                         (df_plot['Model'].isin(['Second Half Excess Time', 'First Half Excess Time']))]
    if len(data_add_2) > 0:
        x_pos = range(len(data_add_2))
        bars = ax3.bar(x_pos, data_add_2['Value'], yerr=[
            data_add_2['Value'] - data_add_2['CI_Lower'],
            data_add_2['CI_Upper'] - data_add_2['Value']
        ], capsize=5, alpha=0.7)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(data_add_2['Model'], rotation=45, ha='right')
        ax3.set_ylabel('Coefficient Value')
        ax3.set_title('Excess Time: Crowd Pressure to Extend Time\n(Home Team Chasing in Close Game)')
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Excess Time models - End Game pressure
    ax4 = axes[1, 1]
    data_end_2 = df_plot[(df_plot['Coefficient'] == 'Crowd Pressure: End Game') & 
                         (df_plot['Model'].isin(['Second Half Excess Time', 'First Half Excess Time']))]
    if len(data_end_2) > 0:
        x_pos = range(len(data_end_2))
        bars = ax4.bar(x_pos, data_end_2['Value'], yerr=[
            data_end_2['Value'] - data_end_2['CI_Lower'],
            data_end_2['CI_Upper'] - data_end_2['Value']
        ], capsize=5, alpha=0.7)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(data_end_2['Model'], rotation=45, ha='right')
        ax4.set_ylabel('Coefficient Value')
        ax4.set_title('Excess Time: Crowd Pressure to End Game\n(Home Team Defending Lead in Close Game)')
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / "plot_bias_comparison.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved bias comparison plot to {output_file}")


def main():
    """Main execution function."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "processed"
    reports_tables_dir = project_root / "reports" / "regression" / "tables"
    reports_figures_dir = project_root / "reports" / "regression" / "figures"
    reports_json_dir = project_root / "reports" / "regression"
    
    # Define train/test seasons
    train_seasons = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
    test_seasons = ["2024-25"]
    
    logger.info("=" * 80)
    logger.info("Starting Regression Analysis")
    logger.info("=" * 80)
    logger.info(f"\nResults will be saved to:")
    logger.info(f"  - Model summaries: {reports_tables_dir}")
    logger.info(f"  - Comparison metrics: {reports_json_dir / 'baseline_comparison_data.json'}")
    logger.info(f"  - Residuals CSV: {reports_tables_dir}")
    logger.info(f"  - Visualizations: {reports_figures_dir}")
    logger.info("=" * 80)
    
    # Step 1: Load and flatten data directly from individual JSON files
    logger.info("\nStep 1: Loading and flattening data from individual JSON files...")
    train_df, test_df = load_and_flatten_data(data_dir, train_seasons, test_seasons)
    
    # Combine for models that use all data
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Step 2: Filter dataframes
    logger.info("\nStep 2: Filtering dataframes...")
    df_baseline_90, df_excess_90, df_placebo_45, df_excess_45 = filter_dataframes(combined_df)
    
    # Step 3: Run Model 1 - Baseline 2nd Half
    logger.info("\nStep 3: Running Model 1 - Baseline 2nd Half...")
    results_model1 = run_baseline_model_90(train_df, test_df, reports_tables_dir)
    
    # Step 4: Run Model 2 - Excess Time 2nd Half
    logger.info("\nStep 4: Running Model 2 - Excess Time 2nd Half...")
    results_model2 = run_excess_model_90(combined_df, reports_tables_dir)
    
    # Step 5: Run Model 3 - Placebo Test 1st Half
    logger.info("\nStep 5: Running Model 3 - Placebo Test 1st Half...")
    results_model3 = run_placebo_test_45(combined_df, reports_tables_dir)
    
    # Step 6: Save comparison metrics JSON
    logger.info("\nStep 6: Saving comparison metrics...")
    results = {
        'model1': results_model1,
        'model2': results_model2,
        'model3': results_model3
    }
    save_comparison_metrics(results, reports_json_dir / "baseline_comparison_data.json")
    
    # Step 7: Create visualizations
    logger.info("\nStep 7: Creating visualizations...")
    plot_bias_comparison(results, reports_figures_dir)
    plot_predicted_vs_actual(results, reports_figures_dir / "plot_predicted_vs_actual.png")
    
    logger.info("\n" + "=" * 80)
    logger.info("Regression Analysis Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

