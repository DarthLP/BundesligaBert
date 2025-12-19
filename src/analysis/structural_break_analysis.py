"""
Structural Break Analysis Script for BundesligaBERT

This script performs rigorous econometric analysis to detect structural breaks in referee behavior
during the 2024/25 Bundesliga season around the policy change date (January 1st, 2025). It compares
BERT predictions against OLS baselines and tests for "Board Effects" (Announced Time inflation)
and "Whistle Effects" (Excess Time constraining).

The analysis includes:
1. Data ingestion and merging BERT predictions with match metadata
2. OLS baseline model refitting on historical data (2018-19 to 2023-24)
3. Comparative metrics (BERT vs OLS) on 2024-25 data
4. Structural break tests: Board Effect and Whistle Effect
5. Interaction regressions to test pressure sensitivity changes

Usage:
    # Run from project root
    python src/analysis/structural_break_analysis.py
    
    # Or as module
    python -m src.analysis.structural_break_analysis

Input:
    - BERT predictions: reports/Bert/tables/bert_predictions_future.csv
    - Processed match data: data/processed/season_2024-25/match_*.json
    - Historical match data: data/processed/season_2018-19/ through season_2023-24/

Output:
    - Historical model outputs: 
      * Summaries: reports/regression_2/tables/summaries/historical_*.txt
      * CSV data: reports/regression_2/tables/csv/historical_residuals_*.csv
    - Comparison tables: reports/regression_2/tables/csv/baseline_comparison_*.csv
    - Structural break tests:
      * CSV data: reports/regression_2/tables/csv/board_effect_*.csv, whistle_effect_*.csv
      * Summaries: reports/regression_2/tables/summaries/board_effect_summary.txt, whistle_effect_summary.txt
    - Interaction regressions: reports/regression_2/tables/summaries/interaction_*.txt
    - Summary files: reports/regression_2/tables/summaries/*_summary.txt
    - JSON metrics: reports/regression_2/*.json
    - Visualizations: reports/regression_2/figures/*.png
      * Interaction effect plots
      * Analysis result visualizations (residuals, time series, test results, comparisons)
    - README: reports/regression_2/README.txt

Author: BundesligaBERT Project
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols

# Import functions from regression_analysis.py
from src.analysis.regression_analysis import (
    load_and_flatten_data,
    engineer_pressure_features,
    filter_dataframes,
    convert_numpy_types,
    extract_score_at_minute
)

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


def load_bert_predictions(bert_predictions_path: Path) -> pd.DataFrame:
    """
    Load BERT predictions from CSV file.
    
    Args:
        bert_predictions_path: Path to bert_predictions_future.csv
        
    Returns:
        DataFrame with BERT predictions
    """
    logger.info(f"Loading BERT predictions from {bert_predictions_path}")
    df = pd.read_csv(bert_predictions_path)
    logger.info(f"Loaded {len(df)} BERT prediction rows")
    return df


def load_2024_25_metadata(data_dir: Path) -> pd.DataFrame:
    """
    Load and construct metadata DataFrame for 2024-25 season.
    
    Args:
        data_dir: Path to data/processed directory
        
    Returns:
        DataFrame with match metadata, features, targets, and flags
    """
    logger.info("Loading 2024-25 season metadata")
    
    season_dir = data_dir / "season_2024-25"
    if not season_dir.exists():
        raise FileNotFoundError(f"Season directory not found: {season_dir}")
    
    matches = []
    match_files = sorted(season_dir.glob('match_*.json'))
    
    for match_file in match_files:
        try:
            with open(match_file, 'r', encoding='utf-8') as f:
                match_data = json.load(f)
            
            # Extract all required fields
            flat_match = {}
            
            # Basic metadata
            flat_match['match_id'] = match_data.get('match_id')
            flat_match['season'] = '2024-25'
            
            metadata = match_data.get('metadata', {})
            flat_match['matchday'] = metadata.get('matchday')
            flat_match['home'] = metadata.get('home')
            flat_match['away'] = metadata.get('away')
            flat_match['attendance'] = metadata.get('attendance')
            
            # Targets
            targets = match_data.get('targets', {})
            flat_match['announced_45'] = targets.get('announced_45')
            flat_match['announced_90'] = targets.get('announced_90')
            flat_match['actual_45'] = targets.get('actual_45')
            flat_match['actual_90'] = targets.get('actual_90')
            flat_match['excess_45'] = targets.get('excess_45')
            flat_match['excess_90'] = targets.get('excess_90')
            
            # Flags
            flags = match_data.get('flags', {})
            flat_match['target_missing_45'] = flags.get('target_missing_45', False)
            flat_match['target_missing_90'] = flags.get('target_missing_90', False)
            flat_match['is_imputed_actual_45'] = flags.get('is_imputed_actual_45', False)
            flat_match['is_imputed_actual_90'] = flags.get('is_imputed_actual_90', False)
            flat_match['is_imputed_announced_45'] = flags.get('is_imputed_announced_45', False)
            flat_match['is_imputed_announced_90'] = flags.get('is_imputed_announced_90', False)
            
            # Features regular
            features_regular = match_data.get('features_regular', {})
            for key, value in features_regular.items():
                flat_match[key] = value
            
            # Features overtime
            features_overtime = match_data.get('features_overtime', {})
            for key, value in features_overtime.items():
                flat_match[key] = value
            
            # Extract scores at 45 minutes from score_timeline for pressure calculation
            score_timeline = match_data.get('score_timeline', {})
            home_score_45, away_score_45 = extract_score_at_minute(score_timeline, 45, inclusive=True)
            flat_match['home_score_45'] = home_score_45
            flat_match['away_score_45'] = away_score_45
            
            matches.append(flat_match)
            
        except Exception as e:
            logger.warning(f"Failed to load {match_file}: {e}")
            continue
    
    df = pd.DataFrame(matches)
    logger.info(f"Loaded {len(df)} matches from 2024-25 season")
    return df


def merge_bert_predictions(metadata_df: pd.DataFrame, bert_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge BERT predictions with metadata DataFrame.
    
    Handles the fact that BERT predictions are in long format (one row per match-half),
    while metadata is in wide format (one row per match).
    
    Args:
        metadata_df: DataFrame with match metadata
        bert_df: DataFrame with BERT predictions
        
    Returns:
        Merged DataFrame with both metadata and BERT predictions
    """
    logger.info("Merging BERT predictions with metadata")
    
    # Create wide format: separate columns for 45 and 90 predictions
    bert_45 = bert_df[bert_df['half'] == 45].copy()
    bert_90 = bert_df[bert_df['half'] == 90].copy()
    
    # Rename columns to distinguish 45 vs 90
    bert_45 = bert_45.rename(columns={
        'actual': 'bert_actual_45',
        'predicted': 'bert_predicted_45',
        'residual': 'bert_residual_45',
        'abs_error': 'bert_abs_error_45'
    })
    
    bert_90 = bert_90.rename(columns={
        'actual': 'bert_actual_90',
        'predicted': 'bert_predicted_90',
        'residual': 'bert_residual_90',
        'abs_error': 'bert_abs_error_90'
    })
    
    # Merge sequentially
    merged = metadata_df.merge(
        bert_45[['match_id', 'bert_actual_45', 'bert_predicted_45', 'bert_residual_45', 'bert_abs_error_45']],
        on='match_id',
        how='left'
    )
    
    merged = merged.merge(
        bert_90[['match_id', 'bert_actual_90', 'bert_predicted_90', 'bert_residual_90', 'bert_abs_error_90']],
        on='match_id',
        how='left'
    )
    
    logger.info(f"Merged DataFrame shape: {merged.shape}")
    logger.info(f"Matches with BERT 45 predictions: {merged['bert_predicted_45'].notna().sum()}")
    logger.info(f"Matches with BERT 90 predictions: {merged['bert_predicted_90'].notna().sum()}")
    
    return merged


def create_structural_break_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create structural break flag based on matchday.
    
    Args:
        df: DataFrame with matchday column
        
    Returns:
        DataFrame with is_ruckrunde column added
    """
    df = df.copy()
    # Matchday < 18: Hinrunde (Baseline) = 0
    # Matchday >= 18: Rückrunde (Treatment) = 1
    df['is_ruckrunde'] = (df['matchday'] >= 18).astype(int)
    logger.info(f"Structural break flag created: {df['is_ruckrunde'].sum()} Rückrunde matches, {len(df) - df['is_ruckrunde'].sum()} Hinrunde matches")
    return df


def filter_for_announced_analysis(df: pd.DataFrame, half: int) -> pd.DataFrame:
    """
    Filter DataFrame for announced time analysis.
    
    Args:
        df: Input DataFrame
        half: 45 or 90
        
    Returns:
        Filtered DataFrame
    """
    if half == 45:
        # Exclude if target_missing_45 == True
        filtered = df[df['target_missing_45'] == False].copy()
    else:  # half == 90
        # Exclude if target_missing_90 == True
        filtered = df[df['target_missing_90'] == False].copy()
    
    return filtered


def filter_for_excess_analysis(df: pd.DataFrame, half: int) -> pd.DataFrame:
    """
    Filter DataFrame for excess time analysis.
    
    Args:
        df: Input DataFrame
        half: 45 or 90
        
    Returns:
        Filtered DataFrame
    """
    if half == 45:
        filtered = df[
            (df['target_missing_45'] == False) &
            (df['is_imputed_actual_45'] == False) &
            (df['is_imputed_announced_45'] == False)
        ].copy()
    else:  # half == 90
        filtered = df[
            (df['target_missing_90'] == False) &
            (df['is_imputed_actual_90'] == False) &
            (df['is_imputed_announced_90'] == False)
        ].copy()
    
    return filtered


def fit_historical_baseline_model(df_train: pd.DataFrame, half: int, output_dir: Path) -> Dict:
    """
    Fit historical baseline model for announced time.
    
    Args:
        df_train: Training DataFrame (historical seasons)
        half: 45 or 90
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with model results
    """
    suffix = str(half)
    
    if half == 45:
        target_col = 'announced_45'
        feature_suffix = '1st'
        formula = (
            f"announced_45 ~ goals_1st + subs_1st + cards_1st + var_1st + "
            f"injuries_1st + disturbances_1st + pressure_add_45 + pressure_draw_45 + pressure_end_45 + "
            f"C(season)"
        )
    else:  # half == 90
        target_col = 'announced_90'
        feature_suffix = '2nd'
        formula = (
            f"announced_90 ~ goals_2nd + subs_2nd + cards_2nd + var_2nd + "
            f"injuries_2nd + disturbances_2nd + pressure_add_90 + pressure_draw_90 + pressure_end_90 + "
            f"C(season)"
        )
    
    logger.info(f"Fitting historical baseline model for {half} (announced)")
    
    # Filter data
    df_filtered = df_train[df_train[target_col].notna()].copy()
    
    # Engineer pressure features
    df_filtered = engineer_pressure_features(df_filtered, minute=half)
    
    # Fit model
    model = ols(formula, data=df_filtered).fit()
    
    # Calculate metrics
    predictions = model.predict(df_filtered)
    actual = df_filtered[target_col]
    train_r2_adj = model.rsquared_adj
    aic = model.aic
    
    # Save summary (use summaries subdirectory)
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    output_file = summaries_dir / f"historical_baseline_{half}_summary.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Historical Baseline Model ({half} minutes - Announced Time)\n")
        f.write("=" * 80 + "\n\n")
        f.write("PURPOSE:\n")
        f.write("Refitting historical models on seasons 2018-19 through 2023-24 for comparison with previous results.\n\n")
        f.write("INPUT DATA:\n")
        f.write("Historical processed match data from data/processed/season_*/\n")
        f.write(f"Filtered according to exclusion criteria: {target_col} not null\n")
        f.write(f"Total samples: {len(df_filtered)}\n\n")
        f.write(f"MODEL FORMULA:\n{formula}\n\n")
        f.write("RESULTS:\n")
        f.write(f"Train R² Adjusted: {train_r2_adj:.4f}\n")
        f.write(f"AIC: {aic:.4f}\n\n")
        f.write("=" * 80 + "\n")
        f.write("Model Summary:\n")
        f.write("=" * 80 + "\n\n")
        f.write(str(model.summary()))
    
    logger.info(f"Saved historical baseline {half} summary to {output_file}")
    
    # Save residuals (use csv subdirectory)
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    residuals_file = csv_dir / f"historical_residuals_baseline_{half}.csv"
    residuals_df = pd.DataFrame({
        'match_id': df_filtered['match_id'].values,
        'actual': actual.values,
        'predicted': predictions.values,
        'residual': (actual - predictions).values,
        'season': df_filtered['season'].values
    })
    residuals_df.to_csv(residuals_file, index=False)
    logger.info(f"Saved residuals to {residuals_file}")
    
    # Extract coefficients
    coefficients = {param: float(model.params[param]) for param in model.params.index}
    p_values = {param: float(model.pvalues[param]) for param in model.pvalues.index}
    
    return {
        'model': model,
        'formula': formula,
        'train_r2_adj': train_r2_adj,
        'aic': aic,
        'n_samples': len(df_filtered),
        'coefficients': coefficients,
        'p_values': p_values,
        'df': df_filtered,
        'predictions': predictions
    }


def fit_historical_excess_model(df_train: pd.DataFrame, half: int, output_dir: Path) -> Dict:
    """
    Fit historical excess model.
    
    Args:
        df_train: Training DataFrame (historical seasons)
        half: 45 or 90
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with model results
    """
    suffix = str(half)
    
    if half == 45:
        target_col = 'excess_45'
        formula = (
            f"excess_45 ~ ot_goals_45 + ot_subs_45 + ot_cards_45 + ot_var_45 + "
            f"ot_injuries_45 + ot_disturbances_45 + pressure_add_45 + pressure_draw_45 + pressure_end_45"
        )
    else:  # half == 90
        target_col = 'excess_90'
        formula = (
            f"excess_90 ~ ot_goals_90 + ot_subs_90 + ot_cards_90 + ot_var_90 + "
            f"ot_injuries_90 + ot_disturbances_90 + pressure_add_90 + pressure_draw_90 + pressure_end_90"
        )
    
    logger.info(f"Fitting historical excess model for {half}")
    
    # Filter data (using filter_dataframes logic - consistent with filter_for_excess_analysis)
    df_filtered = df_train[
        (df_train['target_missing_90' if half == 90 else 'target_missing_45'] == False) &
        (df_train['is_imputed_actual_90' if half == 90 else 'is_imputed_actual_45'] == False) &
        (df_train['is_imputed_announced_90' if half == 90 else 'is_imputed_announced_45'] == False) &
        (df_train[target_col].notna())
    ].copy()
    
    # Also exclude force majeure for 90 (excess_90 > 4.0 minutes)
    if half == 90:
        df_filtered = df_filtered[
            df_filtered['excess_90'] <= 4.0
        ].copy()
    
    # Engineer pressure features
    df_filtered = engineer_pressure_features(df_filtered, minute=half)
    
    # Fit model
    model = ols(formula, data=df_filtered).fit()
    
    # Calculate metrics
    predictions = model.predict(df_filtered)
    actual = df_filtered[target_col]
    train_r2_adj = model.rsquared_adj
    aic = model.aic
    
    # Save summary (use summaries subdirectory)
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    output_file = summaries_dir / f"historical_excess_{half}_summary.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Historical Excess Model ({half} minutes - Excess Time)\n")
        f.write("=" * 80 + "\n\n")
        f.write("PURPOSE:\n")
        f.write("Refitting historical models on seasons 2018-19 through 2023-24 for comparison with previous results.\n\n")
        f.write("INPUT DATA:\n")
        f.write("Historical processed match data from data/processed/season_*/\n")
        f.write(f"Filtered according to exclusion criteria: target_missing_{half} == False, is_imputed_actual_{half} == False, is_imputed_announced_{half} == False, {target_col} not null\n")
        if half == 90:
            f.write("Also excluded force majeure matches: excess_90 > 4.0 minutes\n")
        f.write(f"Total samples: {len(df_filtered)}\n\n")
        f.write(f"MODEL FORMULA:\n{formula}\n\n")
        f.write("RESULTS:\n")
        f.write(f"Train R² Adjusted: {train_r2_adj:.4f}\n")
        f.write(f"AIC: {aic:.4f}\n\n")
        f.write("=" * 80 + "\n")
        f.write("Model Summary:\n")
        f.write("=" * 80 + "\n\n")
        f.write(str(model.summary()))
    
    logger.info(f"Saved historical excess {half} summary to {output_file}")
    
    # Save residuals (use csv subdirectory)
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    residuals_file = csv_dir / f"historical_residuals_excess_{half}.csv"
    residuals_df = pd.DataFrame({
        'match_id': df_filtered['match_id'].values,
        'actual': actual.values,
        'predicted': predictions.values,
        'residual': (actual - predictions).values,
        'season': df_filtered['season'].values
    })
    residuals_df.to_csv(residuals_file, index=False)
    logger.info(f"Saved residuals to {residuals_file}")
    
    # Extract coefficients
    coefficients = {param: float(model.params[param]) for param in model.params.index}
    p_values = {param: float(model.pvalues[param]) for param in model.pvalues.index}
    
    return {
        'model': model,
        'formula': formula,
        'train_r2_adj': train_r2_adj,
        'aic': aic,
        'n_samples': len(df_filtered),
        'coefficients': coefficients,
        'p_values': p_values,
        'df': df_filtered,
        'predictions': predictions
    }


def predict_2024_25_with_baseline_model(model_result: Dict, df_2024_25: pd.DataFrame, half: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Predict on 2024-25 data using baseline model.
    
    Args:
        model_result: Dictionary with 'model' key containing fitted model
        df_2024_25: 2024-25 DataFrame
        half: 45 or 90
        
    Returns:
        Tuple of (predictions array, filtered DataFrame)
    """
    model = model_result['model']
    df = df_2024_25.copy()
    
    # Filter for this half
    if half == 45:
        target_col = 'announced_45'
        df = df[df[target_col].notna()].copy()
        df = engineer_pressure_features(df, minute=45)
    else:
        target_col = 'announced_90'
        df = df[df[target_col].notna()].copy()
        df = engineer_pressure_features(df, minute=90)
    
    # Temporarily set season to 2023-24 for prediction (proxy trick)
    # This is needed because the model was trained on seasons 2018-19 to 2023-24,
    # and the categorical variable C(season) doesn't know how to handle '2024-25'
    df = df.copy()
    df['season'] = '2023-24'
    
    # Drop rows with missing required columns for prediction
    required_cols = []
    if half == 45:
        required_cols = ['goals_1st', 'subs_1st', 'cards_1st', 'var_1st', 'injuries_1st', 
                        'disturbances_1st', 'pressure_add_45', 'pressure_draw_45', 'pressure_end_45', 'season']
    else:
        required_cols = ['goals_2nd', 'subs_2nd', 'cards_2nd', 'var_2nd', 'injuries_2nd', 
                        'disturbances_2nd', 'pressure_add_90', 'pressure_draw_90', 'pressure_end_90', 'season']
    
    df = df.dropna(subset=required_cols)
    
    # Predict
    predictions = model.predict(df)
    return predictions, df


def predict_2024_25_with_excess_model(model_result: Dict, df_2024_25: pd.DataFrame, half: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Predict on 2024-25 data using excess model.
    
    Args:
        model_result: Dictionary with 'model' key containing fitted model
        df_2024_25: 2024-25 DataFrame
        half: 45 or 90
        
    Returns:
        Tuple of (predictions array, filtered DataFrame)
    """
    model = model_result['model']
    
    # Filter using exclusion logic
    df = filter_for_excess_analysis(df_2024_25, half)
    
    # Engineer pressure features
    df = engineer_pressure_features(df, minute=half)
    
    # Get required columns from model formula
    if half == 45:
        required_cols = ['ot_goals_45', 'ot_subs_45', 'ot_cards_45', 'ot_var_45', 
                        'ot_injuries_45', 'ot_disturbances_45', 'pressure_add_45', 
                        'pressure_draw_45', 'pressure_end_45']
    else:  # half == 90
        required_cols = ['ot_goals_90', 'ot_subs_90', 'ot_cards_90', 'ot_var_90', 
                        'ot_injuries_90', 'ot_disturbances_90', 'pressure_add_90', 
                        'pressure_draw_90', 'pressure_end_90']
    
    # Drop rows with missing required columns
    df = df.dropna(subset=required_cols)
    
    # Predict (no season adjustment needed for excess models)
    predictions = model.predict(df)
    return predictions, df


def calculate_comparison_metrics(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    """
    Calculate R² and RMSE for comparison.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Dictionary with r2 and rmse
    """
    # Remove NaN pairs
    mask = actual.notna() & predicted.notna()
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    if len(actual_clean) == 0:
        return {'r2': np.nan, 'rmse': np.nan}
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((actual_clean - predicted_clean) ** 2))
    
    # Calculate R²
    ss_res = np.sum((actual_clean - predicted_clean) ** 2)
    ss_tot = np.sum((actual_clean - np.mean(actual_clean)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    return {'r2': r2, 'rmse': rmse}


def test_board_effect(df_2024_25: pd.DataFrame, half: int, model_type: str, 
                      predicted_col: str, actual_col: str, output_dir: Path) -> Dict:
    """
    Test for Board Effect using calibration approach.
    
    Args:
        df_2024_25: 2024-25 DataFrame with predictions
        half: 45 or 90
        model_type: 'bert' or 'baseline'
        predicted_col: Name of predicted column
        actual_col: Name of actual column
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with test results
    """
    logger.info(f"Testing Board Effect for {model_type} {half}")
    
    # Filter for announced analysis
    df = filter_for_announced_analysis(df_2024_25, half)
    
    # Separate Hinrunde and Rückrunde
    hinrunde = df[df['is_ruckrunde'] == 0].copy()
    ruckrunde = df[df['is_ruckrunde'] == 1].copy()
    
    # Calibrate using Hinrunde
    hinrunde_actual = hinrunde[actual_col].values
    hinrunde_predicted = hinrunde[predicted_col].values
    
    # Remove NaN pairs
    mask = pd.notna(hinrunde_actual) & pd.notna(hinrunde_predicted)
    hinrunde_actual_clean = hinrunde_actual[mask]
    hinrunde_predicted_clean = hinrunde_predicted[mask]
    
    if len(hinrunde_actual_clean) == 0:
        logger.warning(f"No valid Hinrunde data for {model_type} {half}")
        return None
    
    bias = np.mean(hinrunde_actual_clean - hinrunde_predicted_clean)
    
    # Correct Rückrunde predictions
    ruckrunde_actual = ruckrunde[actual_col].values
    ruckrunde_predicted = ruckrunde[predicted_col].values
    
    mask = pd.notna(ruckrunde_actual) & pd.notna(ruckrunde_predicted)
    ruckrunde_actual_clean = ruckrunde_actual[mask]
    ruckrunde_predicted_clean = ruckrunde_predicted[mask]
    
    if len(ruckrunde_actual_clean) == 0:
        logger.warning(f"No valid Rückrunde data for {model_type} {half}")
        return None
    
    corrected_predictions = ruckrunde_predicted_clean + bias
    residuals = ruckrunde_actual_clean - corrected_predictions
    
    # One-sample t-test
    t_stat, p_value = stats.ttest_1samp(residuals, 0)
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals, ddof=1)
    n_obs = len(residuals)
    
    # 95% confidence interval
    ci = stats.t.interval(0.95, n_obs - 1, loc=mean_residual, scale=stats.sem(residuals))
    
    return {
        'model_type': model_type,
        'half': half,
        'calibration_bias': bias,
        'residual_mean': mean_residual,
        'residual_std': std_residual,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'n_obs': n_obs,
        'n_hinrunde': len(hinrunde_actual_clean),
        'n_ruckrunde': n_obs
    }


def test_whistle_effect(df_2024_25: pd.DataFrame, excess_predictions: np.ndarray, 
                        filtered_df: pd.DataFrame, half: int, output_dir: Path) -> Dict:
    """
    Test for Whistle Effect using calibration approach.
    
    Args:
        df_2024_25: 2024-25 DataFrame
        excess_predictions: Predictions from excess model
        filtered_df: Filtered DataFrame (matching predictions)
        half: 45 or 90
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with test results
    """
    logger.info(f"Testing Whistle Effect for {half}")
    
    target_col = f'excess_{half}'
    
    # Separate Hinrunde and Rückrunde
    hinrunde_mask = filtered_df['is_ruckrunde'] == 0
    ruckrunde_mask = filtered_df['is_ruckrunde'] == 1
    
    hinrunde_actual = filtered_df.loc[hinrunde_mask, target_col].values
    hinrunde_predicted = excess_predictions[hinrunde_mask]
    
    ruckrunde_actual = filtered_df.loc[ruckrunde_mask, target_col].values
    ruckrunde_predicted = excess_predictions[ruckrunde_mask]
    
    # Calibrate using Hinrunde
    if len(hinrunde_actual) == 0:
        logger.warning(f"No valid Hinrunde data for excess {half}")
        return None
    
    bias = np.mean(hinrunde_actual - hinrunde_predicted)
    
    # Correct Rückrunde predictions
    if len(ruckrunde_actual) == 0:
        logger.warning(f"No valid Rückrunde data for excess {half}")
        return None
    
    corrected_predictions = ruckrunde_predicted + bias
    residuals = ruckrunde_actual - corrected_predictions
    
    # One-sample t-test
    t_stat, p_value = stats.ttest_1samp(residuals, 0)
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals, ddof=1)
    n_obs = len(residuals)
    
    # 95% confidence interval
    ci = stats.t.interval(0.95, n_obs - 1, loc=mean_residual, scale=stats.sem(residuals))
    
    return {
        'half': half,
        'calibration_bias': bias,
        'residual_mean': mean_residual,
        'residual_std': std_residual,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'n_obs': n_obs,
        'n_hinrunde': len(hinrunde_actual),
        'n_ruckrunde': n_obs
    }


def run_interaction_regression(df_2024_25: pd.DataFrame, half: int, model_type: str, output_dir: Path) -> Dict:
    """
    Run interaction regression to test pressure sensitivity changes.
    
    Args:
        df_2024_25: 2024-25 DataFrame
        half: 45 or 90
        model_type: 'announced' or 'excess'
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with model results
    """
    logger.info(f"Running interaction regression for {model_type} {half}")
    
    if model_type == 'announced':
        if half == 45:
            target_col = 'announced_45'
            feature_suffix = '1st'
            df = filter_for_announced_analysis(df_2024_25, 45)
            df = engineer_pressure_features(df, minute=45)
            formula = (
                f"announced_45 ~ goals_1st + subs_1st + cards_1st + var_1st + "
                f"injuries_1st + disturbances_1st + pressure_add_45 + pressure_draw_45 + pressure_end_45 + "
                f"is_ruckrunde:pressure_add_45 + is_ruckrunde:pressure_draw_45 + is_ruckrunde:pressure_end_45"
            )
        else:  # half == 90
            target_col = 'announced_90'
            feature_suffix = '2nd'
            df = filter_for_announced_analysis(df_2024_25, 90)
            df = engineer_pressure_features(df, minute=90)
            formula = (
                f"announced_90 ~ goals_2nd + subs_2nd + cards_2nd + var_2nd + "
                f"injuries_2nd + disturbances_2nd + pressure_add_90 + pressure_draw_90 + pressure_end_90 + "
                f"is_ruckrunde:pressure_add_90 + is_ruckrunde:pressure_draw_90 + is_ruckrunde:pressure_end_90"
            )
    else:  # model_type == 'excess'
        if half == 45:
            target_col = 'excess_45'
            df = filter_for_excess_analysis(df_2024_25, 45)
            df = engineer_pressure_features(df, minute=45)
            formula = (
                f"excess_45 ~ ot_goals_45 + ot_subs_45 + ot_cards_45 + ot_var_45 + "
                f"ot_injuries_45 + ot_disturbances_45 + pressure_add_45 + pressure_draw_45 + pressure_end_45 + "
                f"is_ruckrunde:pressure_add_45 + is_ruckrunde:pressure_draw_45 + is_ruckrunde:pressure_end_45"
            )
        else:  # half == 90
            target_col = 'excess_90'
            df = filter_for_excess_analysis(df_2024_25, 90)
            df = engineer_pressure_features(df, minute=90)
            formula = (
                f"excess_90 ~ ot_goals_90 + ot_subs_90 + ot_cards_90 + ot_var_90 + "
                f"ot_injuries_90 + ot_disturbances_90 + pressure_add_90 + pressure_draw_90 + pressure_end_90 + "
                f"is_ruckrunde:pressure_add_90 + is_ruckrunde:pressure_draw_90 + is_ruckrunde:pressure_end_90"
            )
    
    # Fit model
    model = ols(formula, data=df).fit()
    
    # Save summary (use summaries subdirectory)
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    output_file = summaries_dir / f"interaction_{model_type}_{half}_summary.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Interaction Regression: {model_type.upper()} {half} minutes\n")
        f.write("=" * 80 + "\n\n")
        f.write("PURPOSE:\n")
        f.write("Test if pressure sensitivity changed after structural break (Rückrunde).\n")
        f.write("Negative interaction coefficients indicate pressure has less impact in Rückrunde.\n\n")
        f.write("INPUT:\n")
        f.write("2024-25 match data with pressure variables and structural break flag (is_ruckrunde).\n")
        f.write(f"Total samples: {len(df)}\n\n")
        f.write(f"MODEL FORMULA:\n{formula}\n\n")
        f.write("=" * 80 + "\n")
        f.write("Model Summary:\n")
        f.write("=" * 80 + "\n\n")
        f.write(str(model.summary()))
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("Interaction Term Coefficients:\n")
        f.write("=" * 80 + "\n\n")
        
        # Extract interaction terms
        interaction_terms = [col for col in model.params.index if 'is_ruckrunde:' in str(col)]
        for term in interaction_terms:
            coef = model.params[term]
            pval = model.pvalues[term]
            f.write(f"{term}: {coef:.4f} (p={pval:.4f})\n")
            f.write(f"  Significant: {'Yes' if pval < 0.05 else 'No'}\n")
            f.write(f"  Interpretation: {'Negative' if coef < 0 else 'Positive'} coefficient indicates pressure has {'less' if coef < 0 else 'more'} impact in Rückrunde\n\n")
    
    logger.info(f"Saved interaction regression summary to {output_file}")
    
    # Extract coefficients
    coefficients = {param: float(model.params[param]) for param in model.params.index}
    p_values = {param: float(model.pvalues[param]) for param in model.pvalues.index}
    
    return {
        'model': model,
        'formula': formula,
        'n_samples': len(df),
        'coefficients': coefficients,
        'p_values': p_values
    }


def run_main_effect_regression(df_2024_25: pd.DataFrame, half: int, model_type: str, 
                               summary_file: Path, output_csv_dir: Path) -> Dict:
    """
    Run main effect regression with is_ruckrunde dummy (no interactions).
    
    Args:
        df_2024_25: 2024-25 DataFrame
        half: 45 or 90
        model_type: 'announced' or 'excess'
        summary_file: Path to summary file to append to (same file as interaction regression)
        output_csv_dir: Directory to save CSV outputs
        
    Returns:
        Dictionary with model results
    """
    logger.info(f"Running main effect regression for {model_type} {half}")
    
    if model_type == 'announced':
        if half == 45:
            target_col = 'announced_45'
            df = filter_for_announced_analysis(df_2024_25, 45)
            df = engineer_pressure_features(df, minute=45)
            formula = (
                f"announced_45 ~ goals_1st + subs_1st + cards_1st + var_1st + "
                f"injuries_1st + disturbances_1st + pressure_add_45 + pressure_draw_45 + pressure_end_45 + "
                f"is_ruckrunde"
            )
        else:  # half == 90
            target_col = 'announced_90'
            df = filter_for_announced_analysis(df_2024_25, 90)
            df = engineer_pressure_features(df, minute=90)
            formula = (
                f"announced_90 ~ goals_2nd + subs_2nd + cards_2nd + var_2nd + "
                f"injuries_2nd + disturbances_2nd + pressure_add_90 + pressure_draw_90 + pressure_end_90 + "
                f"is_ruckrunde"
            )
    else:  # model_type == 'excess'
        if half == 45:
            target_col = 'excess_45'
            df = filter_for_excess_analysis(df_2024_25, 45)
            df = engineer_pressure_features(df, minute=45)
            formula = (
                f"excess_45 ~ ot_goals_45 + ot_subs_45 + ot_cards_45 + ot_var_45 + "
                f"ot_injuries_45 + ot_disturbances_45 + pressure_add_45 + pressure_draw_45 + pressure_end_45 + "
                f"is_ruckrunde"
            )
        else:  # half == 90
            target_col = 'excess_90'
            df = filter_for_excess_analysis(df_2024_25, 90)
            df = engineer_pressure_features(df, minute=90)
            formula = (
                f"excess_90 ~ ot_goals_90 + ot_subs_90 + ot_cards_90 + ot_var_90 + "
                f"ot_injuries_90 + ot_disturbances_90 + pressure_add_90 + pressure_draw_90 + pressure_end_90 + "
                f"is_ruckrunde"
            )
    
    # Fit model
    model = ols(formula, data=df).fit()
    
    # Append summary to existing summary file
    with open(summary_file, 'a', encoding='utf-8') as f:
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("Main Effect Regression (is_ruckrunde dummy only)\n")
        f.write("=" * 80 + "\n\n")
        f.write("PURPOSE:\n")
        f.write("Test for overall structural break effect (intercept shift) between Hinrunde and Rückrunde.\n")
        f.write("This model includes is_ruckrunde as a main effect only (no interactions).\n\n")
        f.write("INPUT:\n")
        f.write("2024-25 match data with pressure variables and structural break flag (is_ruckrunde).\n")
        f.write(f"Total samples: {len(df)}\n\n")
        f.write(f"MODEL FORMULA:\n{formula}\n\n")
        f.write("=" * 80 + "\n")
        f.write("Model Summary:\n")
        f.write("=" * 80 + "\n\n")
        f.write(str(model.summary()))
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("Main Effect Coefficient (is_ruckrunde):\n")
        f.write("=" * 80 + "\n\n")
        
        # Extract is_ruckrunde coefficient
        if 'is_ruckrunde' in model.params.index:
            is_ruckrunde_coef = model.params['is_ruckrunde']
            is_ruckrunde_pval = model.pvalues['is_ruckrunde']
            
            f.write(f"is_ruckrunde: {is_ruckrunde_coef:.4f} (p={is_ruckrunde_pval:.4f})\n")
            f.write(f"  Significant: {'Yes' if is_ruckrunde_pval < 0.05 else 'No'}\n")
            interpretation = "higher" if is_ruckrunde_coef > 0 else "lower"
            f.write(f"  Interpretation: {'Positive' if is_ruckrunde_coef > 0 else 'Negative'} coefficient indicates ")
            f.write(f"{interpretation} {target_col} in Rückrunde compared to Hinrunde\n\n")
    
    logger.info(f"Appended main effect regression summary to {summary_file}")
    
    # Extract all coefficients for CSV
    coefficients = {param: float(model.params[param]) for param in model.params.index}
    p_values = {param: float(model.pvalues[param]) for param in model.pvalues.index}
    
    # Create CSV with coefficients
    csv_data = []
    for param in model.params.index:
        csv_data.append({
            'variable': param,
            'coefficient': float(coefficients[param]),
            'p_value': float(p_values[param]),
            'significant': 'Yes' if p_values[param] < 0.05 else 'No'
        })
    
    csv_df = pd.DataFrame(csv_data)
    csv_file = output_csv_dir / f"main_effect_{model_type}_{half}.csv"
    csv_df.to_csv(csv_file, index=False)
    logger.info(f"Saved main effect regression coefficients to {csv_file}")
    
    return {
        'model': model,
        'formula': formula,
        'n_samples': len(df),
        'coefficients': coefficients,
        'p_values': p_values
    }


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_interaction_effect(df_2024_25: pd.DataFrame, half: int, model_type: str,
                           interaction_result: Dict, output_dir: Path):
    """
    Create interaction effect plot showing pressure sensitivity changes.
    
    Args:
        df_2024_25: DataFrame with predictions and pressure variables
        half: 45 or 90
        model_type: 'announced' or 'excess'
        interaction_result: Dictionary with interaction regression results
        output_dir: Directory to save the figure
    """
    if interaction_result is None:
        return
    
    # Filter and prepare data
    if model_type == 'announced':
        df = filter_for_announced_analysis(df_2024_25, half)
        target_col = f'announced_{half}'
    else:
        df = filter_for_excess_analysis(df_2024_25, half)
        target_col = f'excess_{half}'
    
    df = engineer_pressure_features(df, minute=half)
    
    # Calculate combined pressure variable (average of the three pressure types)
    pressure_cols = [f'pressure_add_{half}', f'pressure_draw_{half}', f'pressure_end_{half}']
    df['pressure_combined'] = df[pressure_cols].mean(axis=1)
    
    # Split by period
    hinrunde = df[df['is_ruckrunde'] == 0].copy()
    ruckrunde = df[df['is_ruckrunde'] == 1].copy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plots
    ax.scatter(hinrunde['pressure_combined'], hinrunde[target_col], 
              alpha=0.5, color='blue', label='Hinrunde', s=30)
    ax.scatter(ruckrunde['pressure_combined'], ruckrunde[target_col],
              alpha=0.5, color='red', label='Rückrunde', s=30)
    
    # Fit regression lines
    if len(hinrunde) > 0:
        hinrunde_clean = hinrunde[['pressure_combined', target_col]].dropna()
        if len(hinrunde_clean) > 1:
            z_hin = np.polyfit(hinrunde_clean['pressure_combined'], hinrunde_clean[target_col], 1)
            p_hin = np.poly1d(z_hin)
            x_hin = np.linspace(hinrunde_clean['pressure_combined'].min(), 
                               hinrunde_clean['pressure_combined'].max(), 100)
            ax.plot(x_hin, p_hin(x_hin), 'b-', linewidth=2, label='Hinrunde Trend')
    
    if len(ruckrunde) > 0:
        ruckrunde_clean = ruckrunde[['pressure_combined', target_col]].dropna()
        if len(ruckrunde_clean) > 1:
            z_ruck = np.polyfit(ruckrunde_clean['pressure_combined'], ruckrunde_clean[target_col], 1)
            p_ruck = np.poly1d(z_ruck)
            x_ruck = np.linspace(ruckrunde_clean['pressure_combined'].min(),
                                ruckrunde_clean['pressure_combined'].max(), 100)
            ax.plot(x_ruck, p_ruck(x_ruck), 'r--', linewidth=2, label='Rückrunde Trend')
    
    ax.set_xlabel('Crowd Pressure (Attendance × Scoreline Tightness)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Added Time (minutes)', fontsize=11, fontweight='bold')
    ax.set_title(f'Interaction Effect: Pressure Sensitivity ({model_type.upper()}, {half} min)', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / f"interaction_effect_{half}_{model_type}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved interaction effect to {output_file}")


def plot_residual_distributions(df_2024_25: pd.DataFrame, half: int, output_dir: Path):
    """
    Create residual distribution comparison plot.
    
    Args:
        df_2024_25: DataFrame with predictions and actuals
        half: 45 or 90
        output_dir: Directory to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Filter data
    df = filter_for_announced_analysis(df_2024_25, half)
    hinrunde = df[df['is_ruckrunde'] == 0].copy()
    ruckrunde = df[df['is_ruckrunde'] == 1].copy()
    
    actual_col = f'announced_{half}'
    
    # BERT residuals
    for period, period_df, ax_idx, color, label in [
        (hinrunde, hinrunde, 0, 'lightblue', 'Hinrunde'),
        (ruckrunde, ruckrunde, 0, 'lightcoral', 'Rückrunde')
    ]:
        bert_pred = period_df[f'bert_predicted_{half}'].values
        actual = period_df[actual_col].values
        mask = pd.notna(bert_pred) & pd.notna(actual)
        residuals = actual[mask] - bert_pred[mask]
        axes[0].hist(residuals, bins=20, alpha=0.6, color=color, label=label, edgecolor='black')
    
    axes[0].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Residual (Actual - Predicted)', fontsize=10)
    axes[0].set_ylabel('Frequency', fontsize=10)
    axes[0].set_title('BERT Residuals', fontsize=11, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # OLS residuals
    for period, period_df, ax_idx, color, label in [
        (hinrunde, hinrunde, 1, 'lightblue', 'Hinrunde'),
        (ruckrunde, ruckrunde, 1, 'lightcoral', 'Rückrunde')
    ]:
        ols_pred = period_df[f'ols_predicted_{half}'].values
        actual = period_df[actual_col].values
        mask = pd.notna(ols_pred) & pd.notna(actual)
        residuals = actual[mask] - ols_pred[mask]
        axes[1].hist(residuals, bins=20, alpha=0.6, color=color, label=label, edgecolor='black')
    
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Residual (Actual - Predicted)', fontsize=10)
    axes[1].set_ylabel('Frequency', fontsize=10)
    axes[1].set_title('OLS Residuals', fontsize=11, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(f'Residual Distribution Comparison ({half} minutes)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_file = output_dir / f"residual_distributions_{half}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved residual distributions to {output_file}")


def plot_time_series_predictions(df_2024_25: pd.DataFrame, half: int, output_dir: Path):
    """
    Create time series plot of predictions vs actual over matchdays.
    
    Args:
        df_2024_25: DataFrame with predictions and actuals
        half: 45 or 90
        output_dir: Directory to save the figure
    """
    # Filter and sort by matchday
    df = filter_for_announced_analysis(df_2024_25, half)
    df = df.sort_values('matchday').copy()
    
    actual_col = f'announced_{half}'
    bert_pred_col = f'bert_predicted_{half}'
    ols_pred_col = f'ols_predicted_{half}'
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot lines
    mask = df[actual_col].notna()
    ax.plot(df.loc[mask, 'matchday'], df.loc[mask, actual_col], 
           'o-', label='Actual', color='black', linewidth=2, markersize=4)
    
    mask = df[bert_pred_col].notna()
    ax.plot(df.loc[mask, 'matchday'], df.loc[mask, bert_pred_col],
           's-', label='BERT Predicted', color='blue', linewidth=1.5, markersize=3, alpha=0.7)
    
    mask = df[ols_pred_col].notna()
    ax.plot(df.loc[mask, 'matchday'], df.loc[mask, ols_pred_col],
           '^-', label='OLS Predicted', color='red', linewidth=1.5, markersize=3, alpha=0.7)
    
    # Add vertical line for structural break
    ax.axvline(x=18, color='green', linestyle='--', linewidth=2, 
               label='Structural Break (Matchday 18)')
    
    ax.set_xlabel('Matchday', fontsize=11, fontweight='bold')
    ax.set_ylabel('Announced Time (minutes)', fontsize=11, fontweight='bold')
    ax.set_title(f'Time Series: Predictions vs Actual ({half} minutes)', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / f"time_series_predictions_{half}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved time series predictions to {output_file}")


def plot_board_effect_results(board_effect_results: Dict, output_dir: Path):
    """
    Create bar chart showing Board Effect test results.
    
    Args:
        board_effect_results: Dictionary with Board Effect test results
        output_dir: Directory to save the figure
    """
    # Prepare data
    results_list = []
    for half in [45, 90]:
        for model_type in ['bert', 'baseline']:
            result = board_effect_results.get(half, {}).get(model_type)
            if result is not None:
                results_list.append({
                    'half': half,
                    'model': model_type.upper(),
                    'mean': result['residual_mean'],
                    'ci_lower': result['ci_lower'],
                    'ci_upper': result['ci_upper'],
                    'p_value': result['p_value']
                })
    
    if not results_list:
        return
    
    df_results = pd.DataFrame(results_list)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create grouped bar chart
    x_pos = np.arange(len(df_results))
    width = 0.35
    
    bars = ax.bar(x_pos, df_results['mean'], width, yerr=[
        df_results['mean'] - df_results['ci_lower'],
        df_results['ci_upper'] - df_results['mean']
    ], capsize=5, alpha=0.7, edgecolor='black')
    
    # Color bars by significance
    colors = ['green' if p < 0.05 else 'gray' for p in df_results['p_value']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add labels
    labels = [f"{row['model']}\n{row['half']}min" for _, row in df_results.iterrows()]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Residual Mean (with 95% CI)', fontsize=11, fontweight='bold')
    ax.set_title('Board Effect Test Results\n(Significant if p < 0.05)', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add p-value annotations
    for i, (_, row) in enumerate(df_results.iterrows()):
        sig_text = f"p={row['p_value']:.3f}"
        ax.text(i, row['mean'] + (row['ci_upper'] - row['mean']) + 0.1, sig_text,
               ha='center', fontsize=9)
    
    plt.tight_layout()
    output_file = output_dir / "board_effect_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved board effect results to {output_file}")


def plot_whistle_effect_results(whistle_effect_results: Dict, output_dir: Path):
    """
    Create bar chart showing Whistle Effect test results.
    
    Args:
        whistle_effect_results: Dictionary with Whistle Effect test results
        output_dir: Directory to save the figure
    """
    # Prepare data
    results_list = []
    for half in [45, 90]:
        result = whistle_effect_results.get(half)
        if result is not None:
            results_list.append({
                'half': half,
                'mean': result['residual_mean'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'p_value': result['p_value']
            })
    
    if not results_list:
        return
    
    df_results = pd.DataFrame(results_list)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    x_pos = np.arange(len(df_results))
    
    bars = ax.bar(x_pos, df_results['mean'], yerr=[
        df_results['mean'] - df_results['ci_lower'],
        df_results['ci_upper'] - df_results['mean']
    ], capsize=5, alpha=0.7, edgecolor='black', color='lightcoral')
    
    # Color bars by significance
    colors = ['red' if p < 0.05 else 'gray' for p in df_results['p_value']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add labels
    labels = [f"{row['half']} minutes" for _, row in df_results.iterrows()]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Residual Mean (with 95% CI)', fontsize=11, fontweight='bold')
    ax.set_title('Whistle Effect Test Results\n(Significant if p < 0.05)', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add p-value annotations
    for i, (_, row) in enumerate(df_results.iterrows()):
        sig_text = f"p={row['p_value']:.3f}"
        ax.text(i, row['mean'] + (row['ci_upper'] - row['mean']) + 0.1, sig_text,
               ha='center', fontsize=10)
    
    plt.tight_layout()
    output_file = output_dir / "whistle_effect_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved whistle effect results to {output_file}")


def plot_model_comparison(comparison_results: Dict, output_dir: Path):
    """
    Create grouped bar chart comparing BERT vs OLS performance.
    
    Args:
        comparison_results: Dictionary with comparison metrics
        output_dir: Directory to save the figure
    """
    # Prepare data
    data = []
    for half in [45, 90]:
        for period in ['hinrunde', 'ruckrunde', 'overall']:
            for model in ['bert', 'ols']:
                metrics = comparison_results.get(half, {}).get(period, {}).get(model, {})
                data.append({
                    'half': half,
                    'period': period.capitalize(),
                    'model': model.upper(),
                    'r2': metrics.get('r2', np.nan),
                    'rmse': metrics.get('rmse', np.nan)
                })
    
    df = pd.DataFrame(data)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # R² and RMSE comparison for each half
    for row_idx, metric in enumerate(['r2', 'rmse']):
        for col_idx, half in enumerate([45, 90]):
            ax = axes[row_idx, col_idx]
            half_df = df[df['half'] == half]
            
            x = np.arange(len(half_df['period'].unique()))
            width = 0.35
            
            bert_vals = half_df[half_df['model'] == 'BERT'][metric].values
            ols_vals = half_df[half_df['model'] == 'OLS'][metric].values
            periods = half_df['period'].unique()
            
            bars1 = ax.bar(x - width/2, bert_vals, width, label='BERT', alpha=0.7, color='blue', edgecolor='black')
            bars2 = ax.bar(x + width/2, ols_vals, width, label='OLS', alpha=0.7, color='red', edgecolor='black')
            
            ax.set_xlabel('Period', fontsize=10)
            metric_label = 'R²' if metric == 'r2' else 'RMSE'
            ax.set_ylabel(metric_label, fontsize=10)
            ax.set_title(f'{metric_label} Comparison ({half} minutes)', fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(periods, fontsize=9)
            if row_idx == 0 and col_idx == 1:  # Only show legend once
                ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('BERT vs OLS Performance Comparison', fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    output_file = output_dir / "model_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved model comparison to {output_file}")


def plot_interaction_coefficients(interaction_results: Dict, output_dir: Path):
    """
    Create heatmap of interaction coefficients.
    
    Args:
        interaction_results: Dictionary with interaction regression results
        output_dir: Directory to save the figure
    """
    # Extract interaction terms
    data = []
    for model_key, result in interaction_results.items():
        model_type, half_str = model_key.split('_')
        half = int(half_str)
        
        model = result['model']
        interaction_terms = [col for col in model.params.index if 'is_ruckrunde:' in str(col)]
        
        for term in interaction_terms:
            coef = float(model.params[term])
            pval = float(model.pvalues[term])
            pressure_var = term.split(':')[1]
            
            data.append({
                'model': f"{model_type}_{half}",
                'pressure_var': pressure_var,
                'coefficient': coef,
                'p_value': pval,
                'significant': pval < 0.05
            })
    
    if not data:
        return
    
    df = pd.DataFrame(data)
    
    # Create pivot table for heatmap
    pivot = df.pivot_table(values='coefficient', index='pressure_var', columns='model')
    significance = df.pivot_table(values='significant', index='pressure_var', columns='model')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create heatmap
    im = ax.imshow(pivot.values, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot.index)
    
    # Add text annotations with significance markers
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            text = f"{pivot.iloc[i, j]:.3f}"
            if significance.iloc[i, j]:
                text += "*"
            ax.text(j, i, text, ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Interaction Coefficients Heatmap\n(* indicates p < 0.05)', 
                fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Coefficient Value')
    
    plt.tight_layout()
    output_file = output_dir / "interaction_coefficients_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved interaction coefficients heatmap to {output_file}")


def main():
    """Main execution function."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "processed"
    bert_predictions_path = project_root / "reports" / "Bert" / "tables" / "bert_predictions_future.csv"
    output_tables_dir = project_root / "reports" / "regression_2" / "tables"
    output_csv_dir = output_tables_dir / "csv"
    output_summaries_dir = output_tables_dir / "summaries"
    output_json_dir = project_root / "reports" / "regression_2"
    output_figures_dir = project_root / "reports" / "regression_2" / "figures"
    
    # Create output directories
    output_tables_dir.mkdir(parents=True, exist_ok=True)
    output_csv_dir.mkdir(parents=True, exist_ok=True)
    output_summaries_dir.mkdir(parents=True, exist_ok=True)
    output_json_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Starting Structural Break Analysis")
    logger.info("=" * 80)
    
    # ========================================================================
    # Module 1: Data Ingestion & "The Golden Join"
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Module 1: Data Ingestion & The Golden Join")
    logger.info("=" * 80)
    
    # Load BERT predictions
    bert_df = load_bert_predictions(bert_predictions_path)
    
    # Load 2024-25 metadata
    metadata_df = load_2024_25_metadata(data_dir)
    
    # Calculate pressure metrics
    metadata_df = engineer_pressure_features(metadata_df, minute=45)
    metadata_df = engineer_pressure_features(metadata_df, minute=90)
    
    # Create structural break flag
    metadata_df = create_structural_break_flag(metadata_df)
    
    # Merge BERT predictions
    df_2024_25 = merge_bert_predictions(metadata_df, bert_df)
    
    logger.info(f"Final merged DataFrame shape: {df_2024_25.shape}")
    
    # ========================================================================
    # Module 3: OLS Baseline Comparison - Fit Historical Models
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Module 3: OLS Baseline Comparison - Fitting Historical Models")
    logger.info("=" * 80)
    
    # Load historical data
    train_seasons = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
    test_seasons = []  # We'll use 2024-25 separately
    train_df, _ = load_and_flatten_data(data_dir, train_seasons, test_seasons)
    
    # Fit 4 historical models
    historical_results = {}
    
    # Baseline 45
    historical_results['baseline_45'] = fit_historical_baseline_model(train_df, 45, output_tables_dir)
    
    # Baseline 90
    historical_results['baseline_90'] = fit_historical_baseline_model(train_df, 90, output_tables_dir)
    
    # Excess 45
    historical_results['excess_45'] = fit_historical_excess_model(train_df, 45, output_tables_dir)
    
    # Excess 90
    historical_results['excess_90'] = fit_historical_excess_model(train_df, 90, output_tables_dir)
    
    # Save coefficients JSON
    coefficients_json = {
        'baseline_45': {
            'coefficients': convert_numpy_types(historical_results['baseline_45']['coefficients']),
            'p_values': convert_numpy_types(historical_results['baseline_45']['p_values'])
        },
        'baseline_90': {
            'coefficients': convert_numpy_types(historical_results['baseline_90']['coefficients']),
            'p_values': convert_numpy_types(historical_results['baseline_90']['p_values'])
        },
        'excess_45': {
            'coefficients': convert_numpy_types(historical_results['excess_45']['coefficients']),
            'p_values': convert_numpy_types(historical_results['excess_45']['p_values'])
        },
        'excess_90': {
            'coefficients': convert_numpy_types(historical_results['excess_90']['coefficients']),
            'p_values': convert_numpy_types(historical_results['excess_90']['p_values'])
        }
    }
    
    with open(output_json_dir / "historical_models_coefficients.json", 'w', encoding='utf-8') as f:
        json.dump(coefficients_json, f, indent=2, ensure_ascii=False)
    
    logger.info("Saved historical model coefficients to JSON")
    
    # ========================================================================
    # Module 3: Predict on 2024-25 and Compute Comparative Metrics
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Module 3: Predicting on 2024-25 and Computing Comparative Metrics")
    logger.info("=" * 80)
    
    # Predict with baseline models
    baseline_45_pred, df_baseline_45_pred = predict_2024_25_with_baseline_model(
        historical_results['baseline_45'], df_2024_25, 45
    )
    baseline_90_pred, df_baseline_90_pred = predict_2024_25_with_baseline_model(
        historical_results['baseline_90'], df_2024_25, 90
    )
    
    # Add predictions to DataFrames
    df_baseline_45_pred['ols_predicted_45'] = baseline_45_pred
    df_baseline_90_pred['ols_predicted_90'] = baseline_90_pred
    
    # Merge back to main DataFrame
    df_2024_25 = df_2024_25.merge(
        df_baseline_45_pred[['match_id', 'ols_predicted_45']],
        on='match_id',
        how='left'
    )
    df_2024_25 = df_2024_25.merge(
        df_baseline_90_pred[['match_id', 'ols_predicted_90']],
        on='match_id',
        how='left'
    )
    
    # Calculate comparison metrics separately for 45 and 90
    comparison_results = {}
    
    for half in [45, 90]:
        actual_col = f'announced_{half}'
        bert_pred_col = f'bert_predicted_{half}'
        ols_pred_col = f'ols_predicted_{half}'
        
        # Filter for announced analysis
        df_half = filter_for_announced_analysis(df_2024_25, half)
        
        # Split by Hinrunde and Rückrunde
        hinrunde = df_half[df_half['is_ruckrunde'] == 0]
        ruckrunde = df_half[df_half['is_ruckrunde'] == 1]
        
        # Calculate metrics
        metrics = {}
        
        # Overall
        bert_metrics_all = calculate_comparison_metrics(df_half[actual_col], df_half[bert_pred_col])
        ols_metrics_all = calculate_comparison_metrics(df_half[actual_col], df_half[ols_pred_col])
        
        # Hinrunde
        bert_metrics_hin = calculate_comparison_metrics(hinrunde[actual_col], hinrunde[bert_pred_col])
        ols_metrics_hin = calculate_comparison_metrics(hinrunde[actual_col], hinrunde[ols_pred_col])
        
        # Rückrunde
        bert_metrics_ruck = calculate_comparison_metrics(ruckrunde[actual_col], ruckrunde[bert_pred_col])
        ols_metrics_ruck = calculate_comparison_metrics(ruckrunde[actual_col], ruckrunde[ols_pred_col])
        
        comparison_results[half] = {
            'overall': {
                'bert': bert_metrics_all,
                'ols': ols_metrics_all
            },
            'hinrunde': {
                'bert': bert_metrics_hin,
                'ols': ols_metrics_hin
            },
            'ruckrunde': {
                'bert': bert_metrics_ruck,
                'ols': ols_metrics_ruck
            }
        }
    
    # Save comparison tables
    for half in [45, 90]:
        comparison_df = pd.DataFrame({
            'period': ['Hinrunde', 'Rückrunde', 'Overall'],
            'bert_r2': [
                comparison_results[half]['hinrunde']['bert']['r2'],
                comparison_results[half]['ruckrunde']['bert']['r2'],
                comparison_results[half]['overall']['bert']['r2']
            ],
            'bert_rmse': [
                comparison_results[half]['hinrunde']['bert']['rmse'],
                comparison_results[half]['ruckrunde']['bert']['rmse'],
                comparison_results[half]['overall']['bert']['rmse']
            ],
            'ols_r2': [
                comparison_results[half]['hinrunde']['ols']['r2'],
                comparison_results[half]['ruckrunde']['ols']['r2'],
                comparison_results[half]['overall']['ols']['r2']
            ],
            'ols_rmse': [
                comparison_results[half]['hinrunde']['ols']['rmse'],
                comparison_results[half]['ruckrunde']['ols']['rmse'],
                comparison_results[half]['overall']['ols']['rmse']
            ]
        })
        
        comparison_df.to_csv(
            output_csv_dir / f"baseline_comparison_announced_{half}_2024_25.csv",
            index=False
        )
    
    # Save comparison summary
    summary_file = output_summaries_dir / "baseline_comparison_2024_25_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BERT vs OLS Baseline Model Comparison (2024-25 Season)\n")
        f.write("=" * 80 + "\n\n")
        f.write("PURPOSE:\n")
        f.write("Compare BERT vs OLS baseline models on 2024-25 season data.\n\n")
        f.write("INPUT:\n")
        f.write("- BERT predictions from bert_predictions_future.csv\n")
        f.write("- OLS predictions from refit historical models (seasons 2018-19 to 2023-24)\n")
        f.write("- 2024-25 match metadata from data/processed/season_2024-25/\n\n")
        f.write("METHODOLOGY:\n")
        f.write("R² and RMSE calculated separately for 45/90 halves and Hinrunde/Rückrunde periods.\n\n")
        f.write("RESULTS:\n\n")
        
        # Include comparison tables
        for half in [45, 90]:
            comparison_df = pd.DataFrame({
                'period': ['Hinrunde', 'Rückrunde', 'Overall'],
                'bert_r2': [
                    comparison_results[half]['hinrunde']['bert']['r2'],
                    comparison_results[half]['ruckrunde']['bert']['r2'],
                    comparison_results[half]['overall']['bert']['r2']
                ],
                'bert_rmse': [
                    comparison_results[half]['hinrunde']['bert']['rmse'],
                    comparison_results[half]['ruckrunde']['bert']['rmse'],
                    comparison_results[half]['overall']['bert']['rmse']
                ],
                'ols_r2': [
                    comparison_results[half]['hinrunde']['ols']['r2'],
                    comparison_results[half]['ruckrunde']['ols']['r2'],
                    comparison_results[half]['overall']['ols']['r2']
                ],
                'ols_rmse': [
                    comparison_results[half]['hinrunde']['ols']['rmse'],
                    comparison_results[half]['ruckrunde']['ols']['rmse'],
                    comparison_results[half]['overall']['ols']['rmse']
                ]
            })
            
            # Format float columns
            float_cols = ['bert_r2', 'bert_rmse', 'ols_r2', 'ols_rmse']
            display_df = comparison_df.copy()
            for col in float_cols:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            
            f.write(f"Announced Time ({half} minutes):\n")
            f.write("-" * 80 + "\n")
            f.write(display_df.to_string(index=False))
            f.write("\n\n")
    
    # Save JSON
    comparison_json = convert_numpy_types(comparison_results)
    with open(output_json_dir / "baseline_comparison_2024_25.json", 'w', encoding='utf-8') as f:
        json.dump(comparison_json, f, indent=2, ensure_ascii=False)
    
    logger.info("Saved comparison metrics")
    
    # ========================================================================
    # Module 4: Calibration & Deviation Tests (Structural Break)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Module 4: Calibration & Deviation Tests (Structural Break)")
    logger.info("=" * 80)
    
    # Test A: Board Effect (Announced Time)
    board_effect_results = {}
    
    for half in [45, 90]:
        # BERT
        bert_result = test_board_effect(
            df_2024_25, half, 'bert',
            f'bert_predicted_{half}', f'announced_{half}',
            output_tables_dir
        )
        
        # Baseline OLS
        baseline_result = test_board_effect(
            df_2024_25, half, 'baseline',
            f'ols_predicted_{half}', f'announced_{half}',
            output_tables_dir
        )
        
        board_effect_results[half] = {
            'bert': bert_result,
            'baseline': baseline_result
        }
    
    # Save Board Effect results
    board_effect_summary = []
    for half in [45, 90]:
        for model_type in ['bert', 'baseline']:
            result = board_effect_results[half][model_type]
            if result is not None:
                board_effect_summary.append({
                    'model_type': model_type,
                    'half': half,
                    'calibration_bias': result['calibration_bias'],
                    'residual_mean': result['residual_mean'],
                    'residual_std': result['residual_std'],
                    't_stat': result['t_stat'],
                    'p_value': result['p_value'],
                    'ci_lower': result['ci_lower'],
                    'ci_upper': result['ci_upper'],
                    'n_obs': result['n_obs'],
                    'n_hinrunde': result['n_hinrunde'],
                    'n_ruckrunde': result['n_ruckrunde']
                })
    
    board_effect_df = pd.DataFrame(board_effect_summary)
    
    for half in [45, 90]:
        half_df = board_effect_df[board_effect_df['half'] == half]
        half_df.to_csv(
            output_csv_dir / f"board_effect_announced_{half}.csv",
            index=False
        )
    
    # Save Board Effect summary
    board_summary_file = output_summaries_dir / "board_effect_summary.txt"
    with open(board_summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Board Effect Test Results\n")
        f.write("=" * 80 + "\n\n")
        f.write("PURPOSE:\n")
        f.write("Test for 'Board Effect' - hypothesis that referees inflate Announced Time in Rückrunde\n")
        f.write("to compensate for lost discretion after policy change.\n\n")
        f.write("INPUT:\n")
        f.write("- BERT predictions and OLS baseline predictions on 2024-25 data\n")
        f.write("- Split by Hinrunde (matchday < 18) and Rückrunde (matchday >= 18)\n\n")
        f.write("METHODOLOGY:\n")
        f.write("1. Calibrate bias using Hinrunde: Mean(Actual_Announced - Predicted)\n")
        f.write("2. Correct Rückrunde predictions by adding calibration bias\n")
        f.write("3. Test residuals with one-sample t-test against 0\n\n")
        f.write("INTERPRETATION:\n")
        f.write("Significant positive mean implies Board Effect exists (Actual > Predicted after correction).\n\n")
        f.write("RESULTS:\n\n")
        
        # Include Board Effect tables
        for half in [45, 90]:
            half_df = board_effect_df[board_effect_df['half'] == half]
            if len(half_df) > 0:
                # Select and format columns for readability
                display_df = half_df[['model_type', 'calibration_bias', 'residual_mean', 'residual_std', 
                                     't_stat', 'p_value', 'ci_lower', 'ci_upper', 'n_obs', 'n_hinrunde', 'n_ruckrunde']].copy()
                # Format float columns
                float_cols = ['calibration_bias', 'residual_mean', 'residual_std', 't_stat', 'p_value', 'ci_lower', 'ci_upper']
                for col in float_cols:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                
                f.write(f"Announced Time ({half} minutes):\n")
                f.write("-" * 80 + "\n")
                f.write(display_df.to_string(index=False))
                f.write("\n\n")
    
    logger.info("Saved Board Effect test results")
    
    # Test B: Whistle Effect (Excess Time)
    whistle_effect_results = {}
    
    for half in [45, 90]:
        # Predict with excess model
        excess_pred, filtered_df = predict_2024_25_with_excess_model(
            historical_results[f'excess_{half}'], df_2024_25, half
        )
        
        # Test Whistle Effect
        result = test_whistle_effect(
            df_2024_25, excess_pred, filtered_df, half, output_tables_dir
        )
        
        whistle_effect_results[half] = result
    
    # Save Whistle Effect results
    whistle_summary = []
    for half in [45, 90]:
        result = whistle_effect_results[half]
        if result is not None:
            whistle_summary.append(result)
    
    whistle_df = pd.DataFrame(whistle_summary)
    for half in [45, 90]:
        half_df = whistle_df[whistle_df['half'] == half]
        if len(half_df) > 0:
            half_df.to_csv(
                output_csv_dir / f"whistle_effect_excess_{half}.csv",
                index=False
            )
    
    # Save Whistle Effect summary
    whistle_summary_file = output_summaries_dir / "whistle_effect_summary.txt"
    with open(whistle_summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Whistle Effect Test Results\n")
        f.write("=" * 80 + "\n\n")
        f.write("PURPOSE:\n")
        f.write("Test for 'Whistle Effect' - hypothesis that transparency prevents referees from adding\n")
        f.write("'secret' time, so Excess Time should drop in Rückrunde.\n\n")
        f.write("INPUT:\n")
        f.write("- OLS excess model predictions on 2024-25 data\n")
        f.write("- Split by Hinrunde (matchday < 18) and Rückrunde (matchday >= 18)\n")
        f.write("- Exclusion filters applied: target_missing == False, is_imputed_actual == False, is_imputed_announced == False\n\n")
        f.write("METHODOLOGY:\n")
        f.write("1. Calibrate bias using Hinrunde: Mean(Actual_Excess - OLS_Predicted)\n")
        f.write("2. Correct Rückrunde predictions by adding calibration bias\n")
        f.write("3. Test residuals with one-sample t-test against 0\n\n")
        f.write("INTERPRETATION:\n")
        f.write("Significant negative mean implies Whistle Effect exists (Actual < Predicted after correction).\n\n")
        f.write("RESULTS:\n\n")
        
        # Include Whistle Effect tables
        for half in [45, 90]:
            half_df = whistle_df[whistle_df['half'] == half]
            if len(half_df) > 0:
                # Select and format columns for readability
                display_df = half_df[['half', 'calibration_bias', 'residual_mean', 'residual_std', 
                                     't_stat', 'p_value', 'ci_lower', 'ci_upper', 'n_obs', 'n_hinrunde', 'n_ruckrunde']].copy()
                # Format float columns
                float_cols = ['calibration_bias', 'residual_mean', 'residual_std', 't_stat', 'p_value', 'ci_lower', 'ci_upper']
                for col in float_cols:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                
                f.write(f"Excess Time ({half} minutes):\n")
                f.write("-" * 80 + "\n")
                f.write(display_df.to_string(index=False))
                f.write("\n\n")
    
    logger.info("Saved Whistle Effect test results")
    
    # ========================================================================
    # Module 5: Interaction Regression (Pressure Sensitivity)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Module 5: Interaction Regression (Pressure Sensitivity)")
    logger.info("=" * 80)
    
    interaction_results = {}
    
    # Run interaction regressions
    for model_type in ['announced', 'excess']:
        for half in [45, 90]:
            result = run_interaction_regression(df_2024_25, half, model_type, output_tables_dir)
            interaction_results[f'{model_type}_{half}'] = result
    
    # Run main effect regressions (is_ruckrunde dummy only, no interactions)
    main_effect_results = {}
    
    for model_type in ['announced', 'excess']:
        for half in [45, 90]:
            summary_file = output_summaries_dir / f"interaction_{model_type}_{half}_summary.txt"
            result = run_main_effect_regression(df_2024_25, half, model_type, summary_file, output_csv_dir)
            main_effect_results[f'{model_type}_{half}'] = result
    
    logger.info("Saved main effect regression results")
    
    # Save overall interaction regression summary
    interaction_summary_file = output_summaries_dir / "interaction_regression_summary.txt"
    with open(interaction_summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Interaction Regression Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write("PURPOSE:\n")
        f.write("Test if pressure sensitivity changed after structural break (Rückrunde).\n")
        f.write("Interaction terms (is_ruckrunde * pressure_*) test whether pressure coefficients\n")
        f.write("are different in Rückrunde compared to Hinrunde.\n\n")
        f.write("WHAT THE INTERACTION TERMS TEST:\n")
        f.write("- Negative interaction coefficients: Pressure has LESS impact in Rückrunde\n")
        f.write("- Positive interaction coefficients: Pressure has MORE impact in Rückrunde\n\n")
        f.write("MODELS RUN:\n")
        f.write("- announced_45: Announced time, 1st half\n")
        f.write("- announced_90: Announced time, 2nd half\n")
        f.write("- excess_45: Excess time, 1st half\n")
        f.write("- excess_90: Excess time, 2nd half\n\n")
        f.write("INTERACTION TERM SUMMARY:\n\n")
        
        # Create summary table of interaction terms from all models
        interaction_summary_rows = []
        for model_key, result in interaction_results.items():
            model_type, half_str = model_key.split('_')
            half = int(half_str)
            
            # Extract interaction terms
            model = result['model']
            interaction_terms = [col for col in model.params.index if 'is_ruckrunde:' in str(col)]
            
            for term in interaction_terms:
                coef = float(model.params[term])
                pval = float(model.pvalues[term])
                # Extract pressure variable name (e.g., "pressure_add_45" from "is_ruckrunde:pressure_add_45")
                pressure_var = term.split(':')[1]
                
                interaction_summary_rows.append({
                    'model': f"{model_type}_{half}",
                    'pressure_var': pressure_var,
                    'coefficient': coef,
                    'p_value': pval,
                    'significant': 'Yes' if pval < 0.05 else 'No'
                })
        
        if interaction_summary_rows:
            interaction_summary_df = pd.DataFrame(interaction_summary_rows)
            # Format float columns
            display_df = interaction_summary_df.copy()
            display_df['coefficient'] = display_df['coefficient'].apply(lambda x: f"{x:.4f}")
            display_df['p_value'] = display_df['p_value'].apply(lambda x: f"{x:.4f}")
            
            f.write(display_df.to_string(index=False))
            f.write("\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("See interaction_announced_45_summary.txt, interaction_announced_90_summary.txt,\n")
        f.write("interaction_excess_45_summary.txt, and interaction_excess_90_summary.txt for full model summaries.\n")
    
    logger.info("Saved interaction regression results")
    
    # ========================================================================
    # Generate Interaction Effect Diagrams (Diagram 5)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Generating Interaction Effect Diagrams")
    logger.info("=" * 80)
    
    for model_type in ['announced', 'excess']:
        for half in [45, 90]:
            result = interaction_results.get(f'{model_type}_{half}')
            if result is not None:
                plot_interaction_effect(df_2024_25, half, model_type, result, output_figures_dir)
    
    # ========================================================================
    # Generate Additional Analysis Diagrams (Diagrams 7-12)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Generating Additional Analysis Diagrams")
    logger.info("=" * 80)
    
    # Residual distributions
    for half in [45, 90]:
        plot_residual_distributions(df_2024_25, half, output_figures_dir)
    
    # Time series predictions
    for half in [45, 90]:
        plot_time_series_predictions(df_2024_25, half, output_figures_dir)
    
    # Board Effect results
    plot_board_effect_results(board_effect_results, output_figures_dir)
    
    # Whistle Effect results
    plot_whistle_effect_results(whistle_effect_results, output_figures_dir)
    
    # Model comparison
    plot_model_comparison(comparison_results, output_figures_dir)
    
    # Interaction coefficients heatmap
    plot_interaction_coefficients(interaction_results, output_figures_dir)
    
    # ========================================================================
    # Create README
    # ========================================================================
    readme_file = output_json_dir / "README.txt"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Structural Break Analysis - Output Documentation\n")
        f.write("=" * 80 + "\n\n")
        f.write("PURPOSE:\n")
        f.write("This analysis detects structural breaks in referee behavior during the 2024/25 Bundesliga season\n")
        f.write("around the policy change date (January 1st, 2025). It compares BERT predictions against OLS baselines\n")
        f.write("and tests for 'Board Effects' (Announced Time inflation) and 'Whistle Effects' (Excess Time constraining).\n\n")
        f.write("INPUT DATA SOURCES:\n")
        f.write("- BERT predictions: reports/Bert/tables/bert_predictions_future.csv\n")
        f.write("- 2024-25 match data: data/processed/season_2024-25/match_*.json\n")
        f.write("- Historical match data: data/processed/season_2018-19/ through season_2023-24/\n\n")
        f.write("ANALYSIS MODULES:\n\n")
        f.write("1. Data Ingestion & The Golden Join:\n")
        f.write("   - Merges BERT predictions with match metadata\n")
        f.write("   - Calculates pressure variables\n")
        f.write("   - Creates structural break flag (is_ruckrunde: matchday >= 18)\n\n")
        f.write("2. OLS Baseline Comparison:\n")
        f.write("   - Refits 4 historical models on seasons 2018-19 to 2023-24:\n")
        f.write("     * Baseline 45 (announced_45)\n")
        f.write("     * Baseline 90 (announced_90)\n")
        f.write("     * Excess 45 (excess_45)\n")
        f.write("     * Excess 90 (excess_90)\n")
        f.write("   - Predicts on 2024-25 data\n")
        f.write("   - Compares BERT vs OLS performance (R², RMSE)\n\n")
        f.write("3. Structural Break Tests:\n")
        f.write("   - Board Effect: Tests if referees inflate Announced Time in Rückrunde\n")
        f.write("   - Whistle Effect: Tests if Excess Time drops in Rückrunde\n\n")
        f.write("4. Interaction Regressions:\n")
        f.write("   - Interaction models: Tests if pressure sensitivity changed after structural break\n")
        f.write("     (uses interaction terms: is_ruckrunde * pressure_*)\n")
        f.write("   - Main effect models: Tests for overall structural break effect (intercept shift)\n")
        f.write("     (uses is_ruckrunde as main effect dummy only, no interactions)\n\n")
        f.write("OUTPUT FILE STRUCTURE:\n\n")
        f.write("All outputs are organized in tables/csv/ (for CSV files) and tables/summaries/ (for summary text files).\n\n")
        f.write("Historical Models:\n")
        f.write("  Summaries (tables/summaries/):\n")
        f.write("    - historical_baseline_45_summary.txt, historical_baseline_90_summary.txt\n")
        f.write("    - historical_excess_45_summary.txt, historical_excess_90_summary.txt\n")
        f.write("  CSV data (tables/csv/):\n")
        f.write("    - historical_residuals_baseline_45.csv, historical_residuals_baseline_90.csv\n")
        f.write("    - historical_residuals_excess_45.csv, historical_residuals_excess_90.csv\n")
        f.write("  JSON (root):\n")
        f.write("    - historical_models_coefficients.json\n\n")
        f.write("Comparison Tables:\n")
        f.write("  CSV (tables/csv/):\n")
        f.write("    - baseline_comparison_announced_45_2024_25.csv\n")
        f.write("    - baseline_comparison_announced_90_2024_25.csv\n")
        f.write("  Summaries (tables/summaries/):\n")
        f.write("    - baseline_comparison_2024_25_summary.txt\n")
        f.write("  JSON (root):\n")
        f.write("    - baseline_comparison_2024_25.json\n\n")
        f.write("Structural Break Tests:\n")
        f.write("  CSV (tables/csv/):\n")
        f.write("    - board_effect_announced_45.csv, board_effect_announced_90.csv\n")
        f.write("    - whistle_effect_excess_45.csv, whistle_effect_excess_90.csv\n")
        f.write("  Summaries (tables/summaries/):\n")
        f.write("    - board_effect_summary.txt\n")
        f.write("    - whistle_effect_summary.txt\n\n")
        f.write("Interaction Regressions:\n")
        f.write("  Summaries (tables/summaries/):\n")
        f.write("    - interaction_announced_45_summary.txt, interaction_announced_90_summary.txt\n")
        f.write("      (contains both interaction regression and main effect regression results)\n")
        f.write("    - interaction_excess_45_summary.txt, interaction_excess_90_summary.txt\n")
        f.write("      (contains both interaction regression and main effect regression results)\n")
        f.write("    - interaction_regression_summary.txt\n")
        f.write("  CSV data (tables/csv/):\n")
        f.write("    - main_effect_announced_45.csv, main_effect_announced_90.csv\n")
        f.write("    - main_effect_excess_45.csv, main_effect_excess_90.csv\n")
        f.write("      (coefficients, p-values, and significance flags for main effect regressions)\n\n")
        f.write("Visualizations (figures/):\n")
        f.write("  Interaction Effects:\n")
        f.write("    - interaction_effect_45_announced.png, interaction_effect_90_announced.png\n")
        f.write("    - interaction_effect_45_excess.png, interaction_effect_90_excess.png\n\n")
        f.write("  Analysis Results:\n")
        f.write("    - residual_distributions_45.png, residual_distributions_90.png\n")
        f.write("    - time_series_predictions_45.png, time_series_predictions_90.png\n")
        f.write("    - board_effect_results.png: Board Effect test results with significance\n")
        f.write("    - whistle_effect_results.png: Whistle Effect test results with significance\n")
        f.write("    - model_comparison.png: BERT vs OLS performance comparison\n")
        f.write("    - interaction_coefficients_heatmap.png: Interaction term coefficients\n")
    
    logger.info("Saved README.txt")
    
    logger.info("\n" + "=" * 80)
    logger.info("Structural Break Analysis Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

