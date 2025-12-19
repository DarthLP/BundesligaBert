# Coefficient Comparison: regression_analysis.py vs structural_break_analysis.py

## Summary

The coefficients between `regression_analysis.py` and `structural_break_analysis.py` are **expected to differ** because they use different training datasets. Both are correct; the differences are intentional and methodologically appropriate for their respective purposes.

## Key Differences

### 1. Baseline 90 Model

**regression_analysis.py:**
- **Training seasons**: 2017-18, 2018-19, 2019-20, 2020-21, 2021-22, 2022-23, 2023-24 (7 seasons)
- **Samples**: 761
- **Reference season**: 2017-18 (first season alphabetically)
- **Coefficient example**: Intercept = 1.2082 (relative to 2017-18)

**structural_break_analysis.py:**
- **Training seasons**: 2018-19, 2019-20, 2020-21, 2021-22, 2022-23, 2023-24 (6 seasons)
- **Samples**: 644
- **Reference season**: 2018-19 (first season in the set)
- **Coefficient example**: Intercept = 1.8015 (relative to 2018-19)

**Why different?** `structural_break_analysis.py` excludes 2017-18 to keep the training window consistent and focused on recent history (2018-19 onward). The intercept difference (1.8015 vs 1.2082) is because they're measured relative to different baseline seasons.

### 2. Baseline 45 Model

**regression_analysis.py:**
- **Training data**: ALL seasons INCLUDING 2024-25 (`combined_df`)
- **Samples**: 755
- **Purpose**: Placebo test using all available data
- **Reference**: Includes 2024-25 season coefficient

**structural_break_analysis.py:**
- **Training data**: Only 2018-19 to 2023-24 (excludes 2024-25)
- **Samples**: ~650 (varies by filtering)
- **Purpose**: Historical baseline to compare against 2024-25 predictions

**Why different?** `regression_analysis.py` uses 2024-25 in training for the placebo test (Model 3A), while `structural_break_analysis.py` keeps 2024-25 separate as the test period for structural break detection.

### 3. Excess Models

Both scripts use similar logic for excess models:
- Same formula (no season dummies)
- Same filtering (excludes imputed values, target_missing flags, force majeure for 90min)
- Minor differences in sample sizes due to different training seasons

## Verification: Are Calculations Correct?

✅ **YES, calculations are correct!**

Both scripts:
1. Use the same formula specifications
2. Apply the same filtering logic (target_missing, is_imputed_actual, force majeure)
3. Use the same pressure feature engineering (`engineer_pressure_features`)
4. Use the same OLS regression framework (`statsmodels.formula.api.ols`)

The coefficient differences are **statistically expected** when:
- Training data spans different seasons (different reference categories)
- Different sample sizes (761 vs 644 for baseline_90)
- Different time periods (affects season trends and intercepts)

## Mathematical Verification

For **baseline_90**, the relationship between coefficients should approximately satisfy:
- `intercept_structural + season_2018-19_coef_structural ≈ intercept_regression`
- Where `season_2018-19_coef_structural` is measured relative to 2018-19 (implicitly 0)
- And `intercept_regression` is measured relative to 2017-18

However, because different samples are used (different matches), exact equality is not expected - only approximate relationships should hold.

## Minor Fix Applied

A minor inconsistency was found and fixed:
- **Issue**: `fit_historical_excess_model` did not filter on `is_imputed_announced`, while `filter_for_excess_analysis` (used for 2024-25 predictions) did filter on it
- **Fix**: Added `is_imputed_announced` filter to `fit_historical_excess_model` for consistency between training and prediction
- **Impact**: This ensures that the same filtering criteria are used for both training historical models and making predictions, improving consistency

## Conclusion

✅ **Both scripts are correct**
✅ **Differences are methodologically appropriate**
✅ **No calculation errors detected**
✅ **Minor filtering inconsistency fixed for consistency**

The `structural_break_analysis.py` approach is correct for structural break testing because it:
- Trains on historical data only (2018-19 to 2023-24)
- Keeps 2024-25 as a clean test set
- Allows fair comparison between BERT and OLS predictions on unseen data
- Uses consistent filtering between training and prediction

The `regression_analysis.py` approach is correct for its purpose (placebo testing and general analysis) because it:
- Uses all available data for maximum statistical power
- Includes 2024-25 in training for placebo tests (Model 3A)

