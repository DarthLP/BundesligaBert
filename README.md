# BundesligaBERT: Predicting Stoppage Time

**Repository:** `bundesliga-bert-time`

## Overview
A BERT-based NLP model predicting 1st Bundesliga stoppage time from live ticker text. Investigates referee behavior and home bias by comparing model-predicted norms against actual added time.

**Note:** This project exclusively uses **1st Bundesliga** matches (top tier German football league). It does not include 2nd Bundesliga or 3rd Liga matches.

## Project Goals
1. **Data Engineering:** Scrape Kicker.de for 1st Bundesliga matches (handling strict anti-bot measures) and parse German ticker text.
2. **NLP Modeling:** Fine-tune BERT for regression (predicting minutes).
3. **Economic Analysis:** Calculate "Residuals" (Actual - Predicted) to measure referee bias and perform a Placebo DID analysis on the 2024/25 season.

## Critical Constraints
1. **NO LEAKAGE:** Aggressively remove explicit time announcements (e.g., "Nachspielzeit", "+3") from input text.
2. **Fine-Tuning:** Do NOT train from scratch. Use `BertForSequenceClassification.from_pretrained()`.
3. **Tech Stack:** PyTorch, Hugging Face `transformers`, `pandas`.

## Setup

### Environment
1. Create conda environment: `conda env create -f environment.yml`
2. Activate: `conda activate bundesliga-bert`

### Dependencies
- Python 3.10
- PyTorch, transformers, tokenizers, datasets, accelerate
- pandas, numpy, scikit-learn
- statsmodels (for econometric regression analysis)
- selenium, webdriver-manager (for scraping)
- matplotlib, seaborn (for visualization)
- jupyterlab (for notebooks)

## Project Structure

```
BundesligaBert/
├── data/
│   ├── raw/                    # Scraped match JSON files
│   │   └── season_YYYY-YY/     # Per-season directories
│   ├── processed/              # Processed datasets for training (force majeure matches excluded)
│   ├── removed_from_processed/ # Force majeure matches (excess_time_90 > 3 min) excluded from analysis
│   ├── kaggle_export/          # JSON files exported for Kaggle dataset upload
│   └── results/                # Model outputs and predictions
├── src/
│   ├── data/
│   │   ├── kicker_scraper.py         # Web scraper for Kicker.de
│   │   ├── process_match_data.py     # Process raw data into structured features
│   │   ├── retry_failed_matches.py   # Retry failed match downloads
│   │   └── export_for_kaggle.py      # Export processed data to JSON for Kaggle
│   ├── analysis/
│   │   ├── descriptive_stats.py      # Descriptive statistics and visualizations
│   │   └── diagnose_negative_correlation.py  # Diagnostic analysis
│   ├── features/
│   │   └── preprocessing.py          # Additional text preprocessing (if needed)
│   ├── models/
│   │   ├── train_bert.py             # BERT fine-tuning script (local training)
│   │   └── bert_module.py            # BERT fine-tuning module (TODO)
│   └── visualization/
│       └── plot_results.py          # Visualization utilities (TODO)
├── tests/
│   └── test_scraper_live.py   # Integration tests for scraper
├── notebooks/                  # Jupyter notebooks for analysis
│   └── kaggle_bert_training.ipynb  # Kaggle notebook for GPU-accelerated BERT training
├── reports/
│   ├── processed_data/
│   │   ├── figures/            # Generated plots (19 figures from descriptive_stats.py)
│   │   └── tables/             # Generated tables and reports
│   └── regression/
│       ├── tables/             # Regression model summaries and residuals
│       ├── figures/            # Regression visualizations
│       └── baseline_comparison_data.json  # Comparison metrics JSON
├── environment.yml             # Conda environment specification
└── README.md                  # This file
```

## Data Collection: Simultaneous Season Download

The scraper can download all **1st Bundesliga** match data from 2017-18 to 2024-25 using multiprocessing for parallel execution. This includes only top-tier German football league matches (no 2nd Bundesliga or 3rd Liga matches). **Note:** The dataset does not include any 2025/26 data (not even the first half of the season).

### Quick Start: Download All Seasons

```bash
# From project root directory
python src/data/kicker_scraper.py
```

This will:
- **Run 4 processes simultaneously** (2 seasons per process)
- **Scrape all matchdays** (1-34) for each season
- **Skip already downloaded matches** automatically
- **Save data** to `data/raw/season_{season}/match_{match_id}.json`
- **Show progress** with per-season logging

### How It Works

1. **Multiprocessing**: Each process handles 2 seasons with its own Chrome instance (4 processes total)
2. **Context Manager Pattern**: Uses Python context manager for guaranteed driver cleanup, preventing zombie Chrome processes
3. **Anti-ban measures**:
   - Chrome runs in incognito mode and background (headless)
   - Image loading disabled to save memory
   - Cookies are cleared between requests
   - Random user agents are used
   - Delays of 3-5 seconds between matches (optimized for M1 Mac)
   - Anti-detection flags (`--disable-blink-features=AutomationControlled`, excludes `enable-automation`)
4. **Resume capability**: Already downloaded matches are automatically skipped
5. **Progress tracking**: Each process logs independently with prefixes like `[PROCESS 1: 2024-25, 2023-24]`

### Data Structure

Each match is saved as `data/raw/season_{season}/match_{match_id}.json`:

```json
{
  "match_id": "bayern-gegen-hoffenheim-2024-bundesliga-4862110",
  "metadata": {
    "home_team": "Bayern München",
    "away_team": "TSG Hoffenheim",
    "final_score": "3:0",
    "attendance": 75000,
    "stadium": "Allianz Arena",
    "referee": "Felix Zwayer",
    "kickoff_date": "2024-01-12",
    "kickoff_time": "20:30",
    "is_ghost_game": false
  },
  "targets": {
    "announced_time_45": 2,
    "actual_played_45": 3,
    "announced_time_90": 4,
    "actual_played_90": 6
  },
  "score_timeline": {
    "23": [1, 0],
    "45": [1, 0],
    "67": [2, 0],
    "90": [3, 0]
  },
  "ticker_data": [
    {
      "minute": "23",
      "text": "Tor für Bayern München durch Robert Lewandowski",
      "extracted_score": [1, 0],
      "score_at_event": [1, 0]
    },
    ...
  ]
}
```

### Testing Individual Matches

For testing or debugging, use the test script:

```bash
# Test with dynamic match discovery
python -m tests.test_scraper_live

# Test specific known matches
python -m tests.test_scraper_live --test-known
python -m tests.test_scraper_live --test-bayern
python -m tests.test_scraper_live --test-bremen
```

### Performance Notes

- **Total matches**: ~2,448 matches (8 seasons × ~306 matches/season, 1st Bundesliga only)
- **Estimated time**: Several hours (depends on network speed and delays)
- **Memory usage**: Each process uses ~150-300MB RAM (reduced by disabling image loading)
- **Network**: Requires stable internet connection
- **Processes**: 4 processes (2 seasons per process)

### Troubleshooting

- **If a process fails**: The other processes continue running independently
- **To resume**: Simply run the script again - it will skip already downloaded matches
- **To force re-scrape**: Delete the specific match file or season directory
- **Check logs**: Each process logs independently with clear prefixes

## Scripts

### `src/data/kicker_scraper.py`

**Purpose:** Web scraper for 1st Bundesliga match data from Kicker.de (top tier German football league only)

**Usage:**
```bash
# Download all seasons (multiprocessing)
python src/data/kicker_scraper.py

# Use as a module (Context Manager pattern)
from src.data.kicker_scraper import KickerScraper
from pathlib import Path

with KickerScraper(save_dir=Path("data/raw")) as scraper:
    result = scraper.scrape_full_match(
        match_url="https://www.kicker.de/...",
        season="2023-24",
        matchday=1
    )
```

**Key Features:**
- **Context Manager Pattern**: Guaranteed driver cleanup, prevents zombie Chrome processes
- **Standard Selenium**: Uses `webdriver_manager` for automatic ChromeDriver management (more stable than undetected_chromedriver)
- Multi-tab scraping (Spielinfo + Ticker)
- Automatic leakage removal (removes "Nachspielzeit" announcements from text)
- Extracts targets (announced_time, actual_played_time) before cleaning text
- Handles JavaScript-rendered content with Selenium
- **Memory optimization**: Disables image loading to save RAM
- Anti-ban measures (incognito mode, random delays, user agents, anti-detection flags)

**Output:** JSON files in `data/raw/season_{season}/match_{match_id}.json`

### `tests/test_scraper_live.py`

**Purpose:** Integration tests for the KickerScraper

**Usage:**
```bash
# Dynamic match discovery (default)
python -m tests.test_scraper_live

# Test known matches
python -m tests.test_scraper_live --test-known
python -m tests.test_scraper_live --test-bayern
python -m tests.test_scraper_live --test-bremen
```

**Features:**
- Dynamically discovers match URLs
- Tests structure, targets, leakage detection, and data quality
- Saves results to `tests/artifacts/smoke_test_result.json`

### `src/data/process_match_data.py`

**Purpose:** Process raw match JSON files into structured features for econometric analysis

**Usage:**
```bash
# Process all matches from all seasons (multiprocessing enabled by default)
# Each season runs in a separate process for parallel processing
python src/data/process_match_data.py

# Process matches from a specific season
python src/data/process_match_data.py --season 2022-23

# Process a specific match file
python src/data/process_match_data.py --input data/raw/season_2022-23/match_xxx.json

# Explicitly enable multiprocessing
python src/data/process_match_data.py --multiprocessing

# Disable multiprocessing (for debugging - processes seasons sequentially)
python src/data/process_match_data.py --no-multiprocessing
```

**Multiprocessing:**
- **Default**: Multiprocessing is enabled when processing multiple seasons
- **Parallel Processing**: Each season runs in a separate process, allowing all seasons to process simultaneously
- **Process Isolation**: Each process has its own logging with process-specific prefixes (e.g., `[PROCESS 1: 2023-24]`)
- **Error Resilience**: One season failing doesn't stop others
- **Summary Statistics**: Final summary shows per-season and overall statistics
- **Use Cases**:
  - Enable multiprocessing for faster processing of multiple seasons (default)
  - Disable multiprocessing for debugging or when processing a single season

**Key Features:**
- **Robust Target Extraction** with imputation logic:
  - Preserves raw target values for reference
  - Standard extraction from metadata and regex patterns
  - Intelligent imputation for missing values (5 scenarios)
  - Calculates excess time (Actual - Announced)
  
- **Phase Separation**: Strictly separates events into:
  - Phase 1: 1st Half Regular (minutes 1-45)
  - Phase 2: Halftime Gap (between halves)
  - Phase 3: 2nd Half Regular (minutes 46-90)
  - Phase 4: Overtime (45+ and 90+)

- **Feature Engineering**:
  - Regular features: Goals, Cards, Subs, VAR, Injuries (by half)
  - Overtime features: Same features for overtime periods
  - Corona/Ghost game flags (see Ghost Game Detection below)
  - Score timelines with corrected overtime minutes

**Ghost Game Detection (`is_ghost_game` flag):**
The `is_ghost_game` flag can have three values: `True`, `False`, or `None`.

- **`True`**: Attendance is known and < 1500 (confirmed ghost game)
- **`False`**: Non-corona season with missing attendance (data collection issue, not a ghost game), or attendance is known and ≥ 1500
- **`None`**: Corona season (2019-20, 2020-21, 2021-22) with missing attendance (unknown/uncertain - likely a ghost game but not confirmed)

**Rationale:** During COVID seasons, missing attendance data is more likely to indicate a ghost game (no fans allowed). For other seasons, missing attendance is treated as a data collection issue rather than indicating a ghost game.

- **BERT Input Construction**:
  - Token-based truncation using German BERT tokenizer (`distilbert-base-german-cased`)
  - Smart truncation preserving critical events (goals, cards, subs, VAR, injuries, disturbances)
  - **Truncation Order** (applied when token limit exceeded):
    1. **Replace ALL goals with summaries** - Detects goals by checking `score_timeline` for score changes
       - Preserves team that scored (e.g., `"23: tor (Bayern)"`)
       - First number in score = home team, second = away team
    2. **Replace ALL substitutions with summaries** - Detects substitutions by checking `substitutes` dict
       - Preserves event type (e.g., `"67: wechsel"`)
    3. **Delete non-critical events** - Removes less important events (starting from early period)
    4. **Replace other critical events** - Summarizes cards, VAR, injuries, disturbances (e.g., `"3: gelbe karte"`)
  - **Period Priority** (within each truncation step):
    - Early period events processed first (1-40 min for 1st half, 46-85 min for 2nd half)
    - Late period events preserved as long as possible (41-45 min for 1st half, 86-90 min for 2nd half)
  - **Critical Event Handling**:
    - Critical events are preserved with full details when possible
    - When token limit is exceeded, critical events are replaced with short summaries instead of being removed
    - Short summaries preserve event type and minute
    - For goals, team that scored is determined from score_timeline (first number = home, second = away)
    - Team names are cleaned (removes FC, SC, VfB, etc. prefixes)
    - Logging: All event replacements and deletions are logged to terminal
  - Targets 400 tokens (max 512 with special tokens)
  - Final truncation ensures output ≤ 512 tokens (with special tokens)
  - Failsafe: stops processing after 5 files exceed 512 tokens
  - Token-optimized format:
    - Season token: `[17]` (start year only, e.g., "2017-18" → `[17]`)
    - Teams: `HomeTeam AwayTeam` (no H:/A: labels, first is home)
    - Attendance: `26113` (number only, omitted if None)
    - Corona: `Krank` (only if corona season, omitted otherwise)
    - Score: `Stand 1:0` (for 2nd half only)
    - Events: `20: text` (no brackets, saves tokens)
    - First-half stats removed from second-half input (not important)
  - **Logging**:
    - INFO level: Logs when critical events are replaced with summaries
    - DEBUG level: Logs when non-critical events are deleted
    - Example: `"Match {match_id}: Replacing goal with summary: '23: Müller schießt das 1:0...' -> '23: tor (Bayern)'"`

- **Data Quality Checks**:
  - Validates required cutoff markers (Anpfiff, Halbzeitpfiff, etc.)
  - Verifies score_timeline matches final_score
  - Fixes overtime minutes in timelines (90 → 90+x)

**Output Structure:**
```json
{
  "match_id": "...",
  "season": "2022-23",
  "metadata": {
    "home": "...",
    "away": "...",
    "attendance": 30000,
    "final_score": "2:1",
    "matchday": 1,
    "is_sold_out": false,
    "is_ghost_game": false,
    "is_corona_season": false,
    "stats": {...}
  },
  "targets": {
    "announced_45": 2,
    "actual_45": 3,
    "excess_45": 1,
    "announced_90": 4,
    "actual_90": 5,
    "excess_90": 1,
    "targets_raw": {...}
  },
  "flags": {
    "is_inferred_zero_45": false,
    "target_missing_45": false,
    "is_imputed_actual_45": false,
    "is_imputed_announced_45": false,
    ...
  },
  "features_regular": {
    "goals_1st": 1,
    "cards_1st": 2,
    "subs_1st": 3,
    "var_1st": 0,
    "injuries_1st": 0,
    ...
  },
  "features_overtime": {
    "ot_goals_45": 0,
    "ot_cards_45": 1,
    ...
  },
  "bert_input_45": "[17] Augsburg RB Leipzig 26113 2: Fast das 1:0...",
  "bert_input_90": "[17] Augsburg RB Leipzig 26113 Stand 1:0 46: Spielerwechsel...",
  "bert_input_45_tokens": 283,
  "bert_input_90_tokens": 298,
  "overtime_ticker_45": [...],
  "overtime_ticker_90": [...]
}
```

**Target Imputation Scenarios:**
1. **Missing Board, Short Game**: If Announced is null and Actual ≤ 1 → Set Announced = 0
2. **Missing Board, Long Game**: If Announced is null and Actual > 1 → Keep Announced = null (mark as missing)
3. **Missing Whistle, Valid Board**: If Actual is null and Announced > 0 → Set Actual = Announced (sets `is_imputed_actual`)
4. **Both Missing**: If both are null → Set both = 0
5. **Negative Excess (Announced > Actual)**: If Announced > Actual → Set Actual = Announced (sets `is_imputed_announced`)

**Output:** Processed JSON files in `data/processed/season_{season}/match_{match_id}.json`

### `src/data/retry_failed_matches.py`

**Purpose:** Retry scraping matches that failed during initial download

**Usage:**
```bash
# Retry all failed matches
python src/data/retry_failed_matches.py

# Retry only matches from a specific season
python src/data/retry_failed_matches.py --season 2023-24

# Retry with verbose logging
python src/data/retry_failed_matches.py --verbose
```

**Key Features:**
- Reads failed matches from `data/failed_matches.json`
- Retries scraping with same anti-ban measures as main scraper
- Updates failed_matches.json (removes successfully scraped matches)
- Supports season filtering
- Comprehensive logging

**Output:** Retries scraping and updates `data/failed_matches.json`

### `src/data/remove_force_majeure_matches.py`

**Purpose:** Exclude matches with excess_time_90 > 3 minutes (force majeure events) from analysis

**Usage:**
```bash
# Remove force majeure matches from processed data
python src/data/remove_force_majeure_matches.py
```

**Rationale:**
Matches with excess_time_90 > 4 minutes represent structural breaks (riots, VAR confusion, technical failures, severe injuries) rather than normal referee bias or time-wasting. These must be excluded to preserve statistical validity:

- **OLS Regression Sensitivity**: A single match with 16 minutes of excess time has ~1000x more influence than a normal match (squared error 256 vs 0.25), which can distort regression coefficients and mask subtle bias signals (~30 seconds).
- **BERT Model Generalization**: Rare out-of-distribution events (pitch invasions, police interventions) cannot be generalized by BERT models and cause memorization rather than learning.
- **Normal Bias Range**: Referee bias operates in the -1 to +2 minute range. Excess > 4 minutes indicates external interruptions not captured by model variables (goals, subs, cards).
- **4.0 Minute Threshold**: Matches with exactly 4.0 minutes of excess time are included as they fall within plausible range for high-event matches or VAR reviews.

**Key Features:**
- Moves 10 force majeure matches from `data/processed/` to `data/removed_from_processed/` (only > 4.0 minutes)
- Creates detailed removal log at `data/results/removed_force_majeure_matches.json`
- Matches are automatically filtered out in `regression_analysis.py` and `train_bert.py` (excess_90 > 4.0 filter)

**Threshold:** 4.0 minutes (excludes only matches > 4.0, removes ~0.4% of data while preserving 4.0 minute matches)

**Output:** 
- Files moved to `data/removed_from_processed/season_{season}/match_{match_id}.json`
- Removal log: `data/results/removed_force_majeure_matches.json`

### `src/data/export_for_kaggle.py`

**Purpose:** Export processed match data into JSON files ready for Kaggle dataset upload

**Usage:**
```bash
# Run from project root
python src/data/export_for_kaggle.py

# Or as module
python -m src.data.export_for_kaggle
```

**Key Features:**
- Loads processed matches from `data/processed/` (same logic as `train_bert.py`)
- Creates samples for both halves (45 and 90) from each match
- Splits History data (< 2024-25) into train/val/test_history using 80/10/10 split with match-level splitting
- Separates Future data (2024-25) as test_future
- Uses same random seed (42) and split logic as `train_bert.py` for reproducibility
- Preserves all metadata (match_id, season, half) needed for diagnostics

**Output:** JSON files saved to `data/kaggle_export/`:
- `train.json`: Training samples (80% of History data)
- `val.json`: Validation samples (10% of History data)
- `test_history.json`: Test History samples (10% of History data)
- `test_future.json`: Test Future samples (all 2024-25 season data)

Each JSON file contains an array of objects with structure:
```json
[
  {
    "text": "BERT input text...",
    "label": 2.0,
    "match_id": "match_12345",
    "season": "2023-24",
    "half": 45
  },
  ...
]
```

**Next Steps After Export:**
1. Zip the JSON files: `cd data/kaggle_export && zip bundesliga_bert_data.zip *.json`
2. Upload to Kaggle: Datasets → New Dataset → Upload
3. Use in Kaggle notebook: `/kaggle/input/your-dataset-name/`

### `src/models/train_bert.py`

**Purpose:** BERT fine-tuning script for local training (supports Apple Silicon MPS, CUDA, or CPU)

**Usage:**
```bash
# Run from project root
python src/models/train_bert.py

# Or as module
python -m src.models.train_bert
```

**Key Features:**
- Fine-tunes `distilbert-base-german-cased` for regression (predicting minutes)
- Creates samples for both halves (45 and 90) from each match
- **Force Majeure Filter**: Automatically excludes matches with excess_90 > 4.0 minutes (structural breaks, not bias)
- Four-set split: train, validation, test_history, test_future
- Automatic device detection (MPS > CUDA > CPU)
- Early stopping with patience
- Comprehensive diagnostic plots (10 plots)
- Saves predictions, metrics, and training logs
- **Force Majeure Filter**: Automatically excludes matches with excess_90 > 4.0 minutes (structural breaks, not bias)

**Output:**
- Model checkpoints: `models/checkpoints/bert_unified/`
- Final model: `models/bert_stoppage_time_final/`
- Predictions: `reports/processed_data/bert_predictions_history.csv`, `bert_predictions_future.csv`
- Metrics: `reports/metrics/bert_performance.json`, `bert_training_log.json`
- Diagnostic plots: `reports/figures/bert_diagnostics/*.png` (10 plots)

### `notebooks/kaggle_bert_training.ipynb`

**Purpose:** Complete BERT training pipeline for Kaggle's free GPU environment

**Usage:**
1. **Export Data Locally**: Run `python src/data/export_for_kaggle.py` to create JSON files
2. **Upload to Kaggle**: Zip JSON files and upload as a private Kaggle dataset
3. **Attach Dataset**: In Kaggle notebook, click "Add data" → Search for your dataset → Add it
4. **Enable GPU**: Settings → Accelerator → GPU T4 x2 (or available GPU)
5. **Update Dataset Path**: Change `DATASET_NAME` in the Config cell to your dataset name
6. **Run All**: Run all cells to train the model and generate diagnostics

**Key Features:**
- Self-contained notebook (no local environment needed)
- Staged dependency installation (core first, plotting later)
- Config cell for easy hyperparameter tweaking
- Early data validation (catches issues before training)
- Uses `trainer.predict()` which returns both metrics and predictions
- Evaluates and generates predictions for ALL splits (train, val, test_history, test_future)
- All 10 diagnostic plots from `train_bert.py`
- Comprehensive outputs including aggregated statistics for local comparison

**Outputs** (saved to `/kaggle/working/`):
- Predictions CSV: `bert_predictions_train.csv`, `bert_predictions_val.csv`, `bert_predictions_history.csv`, `bert_predictions_future.csv`
- Metrics JSON: `bert_performance.json`, `bert_training_log.json`, `bert_aggregated_stats.json`
- Diagnostic plots: `figures/*.png` (10 plots)

**Note:** The notebook replicates the exact same training logic as `train_bert.py` but runs on Kaggle's free GPU for faster training.

### `src/analysis/descriptive_stats.py`

**Purpose:** Generate comprehensive descriptive statistics and visualizations for processed match data

**Usage:**
```bash
# Run from project root
python src/analysis/descriptive_stats.py

# Or as module
python -m src.analysis.descriptive_stats
```

**Key Features:**
- **Data Integrity Analysis**: Imputation rates, missing data, ghost games
- **Target Variable Analysis**: Announced vs actual time distributions, time creep analysis
- **Excess Time Analysis**: Distribution of excess time (actual - announced)
- **Event Distributions**: Goals, subs, cards, VAR, injuries, disturbances by half
- **Overtime Analysis**: Overtime event patterns and chaos metrics
- **Bias & Pressure Analysis**: Home bias, attendance effects, crowd pressure

**Output:**
- Figures: `reports/processed_data/figures/*.png` (19 figures)
- Tables: `reports/processed_data/tables/*.csv`
- Report: `reports/processed_data/tables/descriptive_report.txt`

### `src/analysis/diagnose_negative_correlation.py`

**Purpose:** Diagnostic script to investigate negative correlation between goals_2nd/subs_2nd and announced_90

**Usage:**
```bash
python src/analysis/diagnose_negative_correlation.py
```

**Key Features:**
- Analyzes potential causes of negative correlation:
  1. Missing data bias (target_missing_90)
  2. Imputation effects
  3. Selection bias
  4. Conditional correlations by data availability
- Splits analysis by missing data status
- Analyzes extreme cases (high events, low announced time)

**Output:** Console output with diagnostic analysis

### `src/analysis/regression_analysis.py`

**Purpose:** Econometric regression analysis to test hypotheses about crowd pressure effects on referee decisions

**Usage:**
```bash
# Run from project root
python src/analysis/regression_analysis.py

# Or as module
python -m src.analysis.regression_analysis
```

**Key Features:**
- **Force Majeure Filter**: Automatically excludes matches with excess_90 > 4.0 minutes (structural breaks, not bias)
  - Applied during data loading and in all model filters
  - Prevents outliers from distorting regression coefficients
  - Matches with exactly 4.0 minutes are included (plausible range)
  - See `src/data/remove_force_majeure_matches.py` for rationale and details
  
- **Model 1: Baseline 2nd Half** - Predicts `announced_90` using game events and pressure variables
  - Train/test split: 2017-24 train, 2024-25 test
  - Handles unseen season (2024-25) by using 2023-24 season coefficient
  - Calculates RMSE and MAE on test set
  
- **Model 2: Excess Time 2nd Half** - Analyzes `excess_90` (actual - announced) to detect bias
  - Tests significance of crowd pressure coefficients
  - Excludes force majeure matches (excess_90 > 4.0) in filter
  
- **Model 3A: Placebo Baseline 1st Half** - Tests if pressure effects exist in 1st half (should be insignificant)
  - Predicts `announced_45` using 1st half events
  
- **Model 3B: Placebo Excess 1st Half** - Tests if pressure effects exist in 1st half excess time
  - Analyzes `excess_45` (actual - announced) for 1st half

**Pressure Variables:**
- **Crowd Pressure: Extend Time** (`pressure_add`): When home team is losing by 1 in close game (desperate, wants MORE time)
  - Formula: `attendance_norm × home_losing_1 × is_close_game`
  - Expected coefficient: **Positive** (more attendance when losing → more stoppage time)
  
- **Crowd Pressure: Draw** (`pressure_draw`): When game is tied in close game (cautious, may waste time)
  - Formula: `attendance_norm × draw × is_close_game`
  - Expected coefficient: **Unknown** (could be positive or negative depending on behavior)
  
- **Crowd Pressure: End Game** (`pressure_end`): When home team is defending 1-goal lead in close game (wants LESS time)
  - Formula: `attendance_norm × home_defending × is_close_game`
  - Expected coefficient: **Negative** (more attendance when defending → less stoppage time)

**Missing Attendance Handling:**
The regression uses season-specific median imputation for missing attendance data:
- **COVID seasons** (2019-20, 2020-21, 2021-22): Missing attendance → 0 (ghost games, no crowd pressure)
- **Non-COVID seasons**: Missing attendance → season-specific median:
  - 2017-18: 43,250
  - 2018-19: 39,100
  - 2022-23: 45,147
  - 2023-24: 33,305
  - 2024-25: 33,305 (uses 2023-24 median as proxy)

This preserves sample size (20-31% of matches in non-COVID seasons) while using realistic attendance values for pressure calculations.

**Technical Details:**
- Uses `statsmodels.formula.api.ols` for regression
- Separates losing by 1 vs. drawing (avoids canceling out effects)
- Removes `is_blowout` variable (avoids multicollinearity with `close_game`)
- Filters out imputed actual values (prevents attenuation bias)
- Handles unseen seasons in predictions (2024-25 → uses 2023-24 coefficient)

**Output:**
- Model summaries: `reports/regression/tables/regression_*_summary.txt` (4 files)
- Comparison metrics: `reports/regression/baseline_comparison_data.json`
- Residuals CSV: `reports/regression/tables/residuals_*.csv` (4 files)
- Visualizations:
  - `reports/regression/figures/plot_bias_comparison.png` - Pressure coefficient comparison
  - `reports/regression/figures/plot_predicted_vs_actual.png` - Scatter plots for all models

### `src/features/preprocessing.py` (TODO)

**Purpose:** Additional text preprocessing utilities (if needed beyond process_match_data.py)

**Status:** Not yet implemented (most preprocessing is handled in process_match_data.py)

### `src/models/bert_module.py` (TODO)

**Purpose:** BERT fine-tuning module for regression (refactored from train_bert.py)

**Status:** Not yet implemented (functionality exists in `train_bert.py` and Kaggle notebook)

**Planned Features:**
- Refactor training logic from `train_bert.py` into reusable module
- Fine-tune `distilbert-base-german-cased` for regression
- Use `AutoModelForSequenceClassification.from_pretrained()`
- Predict stoppage time from ticker text

### `src/visualization/plot_results.py` (TODO)

**Purpose:** Visualization utilities for results and analysis

**Status:** Not yet implemented

## Data Quality and Filtering

### Force Majeure Filter

Matches with `excess_time_90 > 4 minutes` are excluded from analysis as they represent structural breaks (riots, VAR confusion, technical failures, severe injuries) rather than normal referee bias. Matches with exactly 4.0 minutes are included as they fall within plausible range for high-event matches or VAR reviews. These matches are:

- **Moved** from `data/processed/` to `data/removed_from_processed/` (10 matches total, ~0.4% of data)
- **Automatically filtered** in `regression_analysis.py` and `train_bert.py` (excess_90 > 4.0 threshold)
- **Logged** in `data/results/removed_force_majeure_matches.json` with details

See `src/data/remove_force_majeure_matches.py` for detailed rationale and implementation.

### Data Leakage Prevention

The scraper implements strict leakage prevention:

1. **Extract targets first**: `announced_time_90` and `actual_played_90` are extracted before text cleaning
2. **Remove time announcements**: All sentences containing "Nachspielzeit", "+X", or time patterns are removed
3. **Skip announcement events**: Events that are purely time announcements are excluded from ticker_data
4. **Pattern matching**: Multiple regex patterns catch variations like:
   - "4 Minuten Nachspielzeit"
   - "+4"
   - "Nachspielzeit: 4"
   - "Eine Minute obendrauf"

## Seasons Covered

All seasons contain **1st Bundesliga** matches only (top tier German football league):

- **2017-18, 2018-19**: Pre-Corona, Clean
- **2019-20, 2020-21, 2021-22**: Corona/Ghost Games (flagged but kept)
- **2022-23, 2023-24**: Post-Corona, Clean
- **2024-25**: Placebo Test Season

**Notes:** 
- This project does not include 2nd Bundesliga or 3rd Liga matches.
- **No 2025/26 data** is included (not even the first half of the season). The latest season in this dataset is 2024-25.

## Data Processing Pipeline

1. **Raw Data Collection**: `src/data/kicker_scraper.py` scrapes match data
2. **Data Processing**: `src/data/process_match_data.py` processes raw data into features
3. **Descriptive Analysis**: `src/analysis/descriptive_stats.py` generates statistics and visualizations
4. **Econometric Analysis**: `src/analysis/regression_analysis.py` performs regression analysis with pressure variables
5. **Data Export (Optional)**: `src/data/export_for_kaggle.py` exports JSON files for Kaggle upload
6. **Model Training**: 
   - **Local**: `src/models/train_bert.py` fine-tunes BERT (supports MPS/CUDA/CPU)
   - **Kaggle**: `notebooks/kaggle_bert_training.ipynb` for GPU-accelerated training
7. **Residual Analysis**: Calculate residuals (Actual - Predicted) and perform Placebo DID analysis (TODO)

## Next Steps

1. ✅ **Data Collection**: Web scraper implemented and tested
2. ✅ **Data Processing**: Feature engineering and BERT input construction implemented
3. ✅ **Descriptive Analysis**: Comprehensive statistics and visualizations implemented
4. ✅ **Econometric Analysis**: Regression analysis with pressure variables implemented
5. ✅ **BERT Fine-Tuning**: Training scripts implemented (`train_bert.py` and Kaggle notebook)
6. ✅ **Kaggle Export**: Export script for GPU-accelerated training on Kaggle
7. **Residual Analysis**: Calculate residuals (Actual - Predicted) and perform Placebo DID analysis (TODO)
8. **Model Comparison**: Compare BERT predictions with econometric regression results (TODO)

## License

[Add license information]

## Author

BundesligaBERT Project
