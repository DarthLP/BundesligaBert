================================================================================
Structural Break Analysis - Output Documentation
================================================================================

PURPOSE:
This analysis detects structural breaks in referee behavior during the 2024/25 Bundesliga season
around the policy change date (January 1st, 2025). It compares BERT predictions against OLS baselines
and tests for 'Board Effects' (Announced Time inflation) and 'Whistle Effects' (Excess Time constraining).

INPUT DATA SOURCES:
- BERT predictions: reports/Bert/tables/bert_predictions_future.csv
- 2024-25 match data: data/processed/season_2024-25/match_*.json
- Historical match data: data/processed/season_2018-19/ through season_2023-24/

ANALYSIS MODULES:

1. Data Ingestion & The Golden Join:
   - Merges BERT predictions with match metadata
   - Calculates pressure variables
   - Creates structural break flag (is_ruckrunde: matchday >= 18)

2. OLS Baseline Comparison:
   - Refits 4 historical models on seasons 2018-19 to 2023-24:
     * Baseline 45 (announced_45)
     * Baseline 90 (announced_90)
     * Excess 45 (excess_45)
     * Excess 90 (excess_90)
   - Predicts on 2024-25 data
   - Compares BERT vs OLS performance (R², RMSE)

3. Structural Break Tests:
   - Board Effect: Tests if referees inflate Announced Time in Rückrunde
   - Whistle Effect: Tests if Excess Time drops in Rückrunde

4. Interaction Regressions:
   - Interaction models: Tests if pressure sensitivity changed after structural break
     (uses interaction terms: is_ruckrunde * pressure_*)
   - Main effect models: Tests for overall structural break effect (intercept shift)
     (uses is_ruckrunde as main effect dummy only, no interactions)

OUTPUT FILE STRUCTURE:

All outputs are organized in tables/csv/ (for CSV files) and tables/summaries/ (for summary text files).

Historical Models:
  Summaries (tables/summaries/):
    - historical_baseline_45_summary.txt, historical_baseline_90_summary.txt
    - historical_excess_45_summary.txt, historical_excess_90_summary.txt
  CSV data (tables/csv/):
    - historical_residuals_baseline_45.csv, historical_residuals_baseline_90.csv
    - historical_residuals_excess_45.csv, historical_residuals_excess_90.csv
  JSON (root):
    - historical_models_coefficients.json

Comparison Tables:
  CSV (tables/csv/):
    - baseline_comparison_announced_45_2024_25.csv
    - baseline_comparison_announced_90_2024_25.csv
  Summaries (tables/summaries/):
    - baseline_comparison_2024_25_summary.txt
  JSON (root):
    - baseline_comparison_2024_25.json

Structural Break Tests:
  CSV (tables/csv/):
    - board_effect_announced_45.csv, board_effect_announced_90.csv
    - whistle_effect_excess_45.csv, whistle_effect_excess_90.csv
  Summaries (tables/summaries/):
    - board_effect_summary.txt
    - whistle_effect_summary.txt

Interaction Regressions:
  Summaries (tables/summaries/):
    - interaction_announced_45_summary.txt, interaction_announced_90_summary.txt
      (contains both interaction regression and main effect regression results)
    - interaction_excess_45_summary.txt, interaction_excess_90_summary.txt
      (contains both interaction regression and main effect regression results)
    - interaction_regression_summary.txt
  CSV data (tables/csv/):
    - main_effect_announced_45.csv, main_effect_announced_90.csv
    - main_effect_excess_45.csv, main_effect_excess_90.csv
      (coefficients, p-values, and significance flags for main effect regressions)

Visualizations (figures/):
  Interaction Effects:
    - interaction_effect_45_announced.png, interaction_effect_90_announced.png
    - interaction_effect_45_excess.png, interaction_effect_90_excess.png

  Analysis Results:
    - residual_distributions_45.png, residual_distributions_90.png
    - time_series_predictions_45.png, time_series_predictions_90.png
    - board_effect_results.png: Board Effect test results with significance
    - whistle_effect_results.png: Whistle Effect test results with significance
    - model_comparison.png: BERT vs OLS performance comparison
    - interaction_coefficients_heatmap.png: Interaction term coefficients
